import numpy as np
import argparse
import torch
import torch.nn as nn
import time
import datetime
import os
from models.fast_VQ_VAE import VQ_VAE
from models.model_utils import length_mask,full_mask,loss_mask_func
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn.utils import clip_grad_norm_
from utils import timeSince,setup_seed,data_augment
# from pytorchtools import EarlyStopping

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--Epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--encoder_width", type=int, default=128)
parser.add_argument("--decoder_width", type=int, default=256)
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--transformer_N", type=int, default=8)
parser.add_argument("--multihead_N", type=int, default=8)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--print_every", type=int, default=240)  #250 240
parser.add_argument("--dropout", type=float, default=0.3)
# parser.add_argument("--wight_decay", type=float, default=1e-4)
parser.add_argument("--encoder_attention_type",  type=str, default='linear')
parser.add_argument("--decoder_attention_type",  type=str, default='causal-linear')
parser.add_argument("--cross_attention_type",  type=str, default='full')
parser.add_argument("--activate",  type=str, default='gelu')
parser.add_argument("--data_path",  type=str, default='./data/co-representation/emopia_data.npz')
# parser.add_argument("--data_path",  type=str, default='./data/co-representation/ailabs_data.npz')
parser.add_argument("--model_path",  type=str, default='./saved_models/VQ_VAE/emopia_finetune_encoder_w_aug_wo_emo/')
parser.add_argument("--VQ_VAE_path",  type=str, default='./saved_models/VQ_VAE/ailabs_w_aug_wo_emo_256/VQ_VAE_ailabs_min_0.56.pth')
parser.add_argument("--log_path",  type=str, default='./logs/VQ_VAE_emopia_finetune_encoder_w_aug_wo_emo.txt')


def get_data_loader(train_x,train_y,data_length):
    train_data_x = torch.LongTensor(train_x).to(device)
    train_data_y = torch.LongTensor(train_y).to(device)
    train_data_length = torch.LongTensor(data_length).unsqueeze(-1).to(device)
    train_dataset = TensorDataset(train_data_x, train_data_y, train_data_length)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    return train_loader


def train(input_x,input_y,length,is_train,step,allStep):
    if is_train=="train":
        model.train()
    else:
        model.eval()
    coefficient=1#(1/allStep)*step
    # print(input[0][length[0]-2])
    type, tempo, chord, bar_beat, pitch, duration, velocity, vq_loss, _=model(input_x,input_y)
    # print("input: ",input.shape)
    # print(type.shape, tempo.shape, chord.shape, bar_beat.shape, pitch.shape, duration.shape, velocity.shape)
    type_GT = torch.narrow(input_y, 2, 3, 1).squeeze(-1)
    tempo_GT = torch.narrow(input_y, 2, 0, 1).squeeze(-1)
    chord_GT = torch.narrow(input_y, 2, 1, 1).squeeze(-1)
    bar_beat_GT = torch.narrow(input_y, 2, 2, 1).squeeze(-1)
    pitch_GT = torch.narrow(input_y, 2, 4, 1).squeeze(-1)
    duration_GT = torch.narrow(input_y, 2, 5, 1).squeeze(-1)
    velocity_GT = torch.narrow(input_y, 2, 6, 1).squeeze(-1)
    # print(type_GT.shape, tempo_GT.shape, chord_GT.shape, bar_beat_GT.shape, pitch_GT.shape, duration_GT.shape, velocity_GT.shape)
    type_loss=0; tempo_loss=0; chord_loss=0; bar_beat_loss=0; pitch_loss=0; duration_loss=0; velocity_loss=0
    for i in range(args.batch_size):
        length_i=length[i]
        type_loss += criterion(type[i][:length_i], type_GT[i][:length_i])
        tempo_loss += criterion(tempo[i][:length_i], tempo_GT[i][:length_i])
        chord_loss += criterion(chord[i][:length_i], chord_GT[i][:length_i])
        bar_beat_loss += criterion(bar_beat[i][:length_i], bar_beat_GT[i][:length_i])
        pitch_loss += criterion(pitch[i][:length_i], pitch_GT[i][:length_i])
        duration_loss += criterion(duration[i][:length_i], duration_GT[i][:length_i])
        velocity_loss += criterion(velocity[i][:length_i], velocity_GT[i][:length_i])
    loss = type_loss+tempo_loss+bar_beat_loss+pitch_loss+duration_loss+velocity_loss#+chord_loss
    loss = (loss/args.batch_size) / 6 + coefficient * vq_loss
    # print("loss——type: ",type_loss/ args.batch_size, ", tempo: ",tempo_loss/args.batch_size, ", chord: ",chord_loss/args.batch_size,
    #       ", bar_beat: ",bar_beat_loss/args.batch_size, ", pitch: ",pitch_loss/args.batch_size,", duration: ",duration_loss/args.batch_size,
    #         ", velocity: ",velocity_loss/args.batch_size, "vq: ",vq_loss)
    if is_train == "train":
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
    return loss.item(), type_loss.item()/ args.batch_size,tempo_loss.item()/args.batch_size,chord_loss.item()/args.batch_size, \
            bar_beat_loss.item()/args.batch_size, pitch_loss.item()/args.batch_size, duration_loss.item()/args.batch_size, \
            velocity_loss.item() / args.batch_size, vq_loss.item()

def trainIter():
    max_test_loss=1000
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    epoch_already=-1#VQ_VAE_dict['epoch']
    step=0
    allStep=(train_data_num//args.batch_size)*100
    for epoch in range(epoch_already+1, args.Epoch):
        f = open(args.log_path, 'a')
        # 数据划分
        print("-----------------------------epoch ", epoch, "------------------------------")
        f.write('-----------------------------epoch %d------------------------------\n' % (epoch))
        total_loss = 0
        train_total_loss = 0
        type_loss_total = 0;tempo_loss_total = 0;chord_loss_total = 0;bar_beat_loss_total = 0;
        pitch_loss_total = 0;duration_loss_total = 0;velocity_loss_total = 0;vq_loss_total=0
        for i, data in enumerate(train_loader):
            input_x,input_y, length = data
            # print(input.shape,length.shape)
            loss,type_loss,tempo_loss,chord_loss,bar_beat_loss,pitch_loss,duration_loss,velocity_loss,vq_loss\
                = train(input_x,input_y,length,"train",step,allStep)
            step+=1
            # print("losses: ",loss,type_loss,tempo_loss,chord_loss,bar_beat_loss,pitch_loss,duration_loss,velocity_loss,vq_loss)
            total_loss += loss
            train_total_loss += loss
            type_loss_total+=type_loss;tempo_loss_total+=tempo_loss;chord_loss_total+=chord_loss;bar_beat_loss_total+=bar_beat_loss
            pitch_loss_total+=pitch_loss;duration_loss_total+=duration_loss;velocity_loss_total+=velocity_loss;vq_loss_total+=vq_loss
            if (i+1) % (args.print_every) == 0:
                print('epoch train:%d, %s(%d %d%%) %.10f' % (
                    epoch, timeSince(start_time), i+1, (i+1) / (train_data_num/args.batch_size) * 100, total_loss / args.print_every))
                f.write('epoch train:%d, %s(%d %d%%) %.10f\n' % (
                    epoch, timeSince(start_time), i+1, (i+1) / (train_data_num/args.batch_size) * 100, total_loss / args.print_every))
                print("--------------------------------------------------------------")
                total_loss = 0
        eval_total_loss=0
        type_eval_total = 0; tempo_eval_total = 0; chord_eval_total = 0; bar_beat_eval_total = 0;
        pitch_eval_total = 0; duration_eval_total = 0; velocity_eval_total = 0; vq_eval_total = 0
        for j,data in enumerate(eval_loader):
            input_x, input_y, length = data
            # print(input.shape,length.shape)
            loss, type_loss, tempo_loss, chord_loss, bar_beat_loss, pitch_loss, duration_loss, velocity_loss, vq_loss \
                = train(input_x, input_y, length, "eval", step, allStep)
            type_eval_total+=type_loss;tempo_eval_total+=tempo_loss;chord_eval_total+=chord_loss;bar_beat_eval_total+=bar_beat_loss
            pitch_eval_total+=pitch_loss;duration_eval_total+=duration_loss;velocity_eval_total+=velocity_loss;vq_eval_total+=vq_loss
            eval_total_loss+=loss
        print('epoch: %d, time: %s, \ntrain_type_loss: %.6f, eval_type_loss: %.6f, \ntrain_tempo_loss: %.6f, eval_tempo_loss: %.6f, \n'
              'train_chord_loss: %.6f, eval_chord_loss: %.6f, \ntrain_bar_beat_loss: %.6f, eval_bar_beat_loss: %.6f, \n'
              'train_pitch_loss: %.6f, eval_pitch_loss: %.6f, \ntrain_duration_loss: %.6f, eval_duration_loss: %.6f, \n'
              'train_velocity_loss: %.6f, eval_velocity_loss: %.6f, \ntrain_vq_loss: %.10f, eval_vq_loss: %.6f, \ntrain loss: %.6f, eval loss: %.6f'
              % (epoch, timeSince(start_time), type_loss_total/(i),type_eval_total/(j),tempo_loss_total/(i),tempo_eval_total/(j),
                 chord_loss_total/(i),chord_eval_total/(j),bar_beat_loss_total/(i),bar_beat_eval_total/(j),
                 pitch_loss_total/(i),pitch_eval_total/(j),duration_loss_total/(i),duration_eval_total/(j),
                 velocity_loss_total/(i),velocity_eval_total/(j),vq_loss_total/(i),vq_eval_total/(j),train_total_loss / (i), eval_total_loss/(j)))
        f.write('epoch: %d, time: %s, \ntrain_type_loss: %.6f, eval_type_loss: %.6f, \ntrain_tempo_loss: %.6f, eval_tempo_loss: %.6f, \n'
              'train_chord_loss: %.6f, eval_chord_loss: %.6f, \ntrain_bar_beat_loss: %.6f, eval_bar_beat_loss: %.6f, \n'
              'train_pitch_loss: %.6f, eval_pitch_loss: %.6f, \ntrain_duration_loss: %.6f, eval_duration_loss: %.6f, \n'
              'train_velocity_loss: %.6f, eval_velocity_loss: %.6f, \ntrain_vq_loss: %.10f, eval_vq_loss: %.6f, \ntrain loss: %.6f, eval loss: %.6f\n'
              % (epoch, timeSince(start_time), type_loss_total/(i),type_eval_total/(j),tempo_loss_total/(i),tempo_eval_total/(j),
                 chord_loss_total/(i),chord_eval_total/(j),bar_beat_loss_total/(i),bar_beat_eval_total/(j),
                 pitch_loss_total/(i),pitch_eval_total/(j),duration_loss_total/(i),duration_eval_total/(j),
                 velocity_loss_total/(i),velocity_eval_total/(j),vq_loss_total/(i),vq_eval_total/(j),train_total_loss / (i), eval_total_loss/(j)))
        train_average=train_total_loss/(i)
        eval_average=eval_total_loss/(j)
        if eval_average < max_test_loss:
            print("epoch: %d save min test loss model-->test loss: %.6f" % (epoch, eval_average))
            f.write('epoch: %d save min test loss model-->test loss: %.6f\n' % (epoch, eval_average))
            if eval_average > 0.7:
                model_save_path = args.model_path + "VQ_VAE_ailabs_high_loss.pth"
            else:
                loss_temp = round(eval_average * 100)
                model_save_path = args.model_path + "VQ_VAE_ailabs_min_0." + str(loss_temp) + ".pth"
            # min_loss_path=args.model_path+"VQ_VAE_emopia_finetune_min_loss.pth"
                # model_save_path = args.model_path + "VQ_VAE_ailabs_pretrain_" + date + "_epoch" + str(
                #     epoch) + "_min_" + str(round(train_average, 4)) + ".pth"
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_save_path)
            # torch.save(state, min_loss_path)
            max_test_loss = eval_average
        f.close()


if __name__=='__main__':
    setup_seed(13)
    args = parser.parse_args()
    f = open(args.log_path, 'a')
    #f.write("emotion as the first token for both encoder and decoder, VQ embeddings concat emotion embeddings\n")
    f.write("--------------------------------------------------\n")
    for arg in vars(args):
        f.write(str(arg)+"="+str(getattr(args,arg))+"\n")
    f.close()
    data = np.load(args.data_path)
    train_x=data['x']
    train_y=data['y']
    data_length = data['seq_len']
    print("data_shape: ", train_x.shape, train_y.shape, data_length.shape)
    print("data_num: ", len(train_x))
    train_x_aug,train_y_aug,train_length_aug=data_augment(train_x[:960],train_y[:960],data_length[:960])
    eval_x_aug,eval_y_aug,eval_length_aug=data_augment(train_x[960:],train_y[960:],data_length[960:])
    assert len(train_x_aug)==len(train_y_aug)==len(train_length_aug)
    assert len(eval_x_aug) == len(eval_y_aug) == len(eval_length_aug)
    train_data_num = len(train_x_aug); eval_data_num = len(eval_x_aug)
    print("train num: ",train_data_num,", eval num: ",eval_data_num)
    train_loader=get_data_loader(train_x_aug,train_y_aug,train_length_aug)
    eval_loader=get_data_loader(eval_x_aug,eval_y_aug,eval_length_aug)
    VQ_VAE_model=VQ_VAE(args.transformer_N, args.multihead_N, args.encoder_width, args.decoder_width,
                 args.n_embeddings, args.embedding_dim, args.dropout, args.activate,
                 args.encoder_attention_type,args.decoder_attention_type,).to(device)
    #model_dict=VQ_VAE_model.state_dict()
    resume=args.VQ_VAE_path
    VQ_VAE_dict=torch.load(resume,map_location=device)
    #pretrain_dict = {k: v for k,v in VQ_VAE_dict['model'].items() if not \
    #    (k.find("transformer_decoder")!=-1 or k.find("project")!=-1 or k.find("predict")!=-1 or k.find("decoder_linear")!=-1)}
    #model_dict.update(pretrain_dict)
    #VQ_VAE_model.load_state_dict(model_dict)
    VQ_VAE_model.load_state_dict(VQ_VAE_dict['model'])
    # VQ_VAE_model.VQ.embeddings=torch.empty((args.n_embeddings, args.embedding_dim)).to(device)
    # nn.init.xavier_uniform_(VQ_VAE_model.VQ.embeddings)
    model=VQ_VAE_model
    criterion = nn.NLLLoss().to(device)
    # criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)#, weight_decay=1e-4)
    # optimizer.load_state_dict(VQ_VAE_dict['optimizer'])
    start_time = time.time()
    date= str(datetime.date.today())
    trainIter()






