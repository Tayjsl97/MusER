"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import numpy as np
import argparse
import torch
import torch.nn as nn
import time
import os
from models.MusER_TRANS_CA_GE import VQ_VAE
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils import clip_grad_norm_
from utils import timeSince,setup_seed


def get_data_loader(train_x,train_y,data_length):
    train_data_x = torch.LongTensor(train_x).to(device)
    train_data_y = torch.LongTensor(train_y).to(device)
    train_data_length = torch.LongTensor(data_length).unsqueeze(-1).to(device)
    train_dataset = TensorDataset(train_data_x, train_data_y, train_data_length)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader


def train(input_x,input_y,length,is_train):
    if is_train=="train":
        model.train()
    else:
        model.eval()
    type, tempo, chord, bar_beat, pitch, duration, velocity, vq_loss, attr_loss\
        = model(input_x, input_y)
    type_GT = torch.narrow(input_y, 2, 3, 1).squeeze(-1)
    tempo_GT = torch.narrow(input_y, 2, 0, 1).squeeze(-1)
    chord_GT = torch.narrow(input_y, 2, 1, 1).squeeze(-1)
    bar_beat_GT = torch.narrow(input_y, 2, 2, 1).squeeze(-1)
    pitch_GT = torch.narrow(input_y, 2, 4, 1).squeeze(-1)
    duration_GT = torch.narrow(input_y, 2, 5, 1).squeeze(-1)
    velocity_GT = torch.narrow(input_y, 2, 6, 1).squeeze(-1)
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
    real_loss=(type_loss+tempo_loss+duration_loss+velocity_loss+chord_loss+bar_beat_loss + pitch_loss)/(7*args.batch_size)
    loss=real_loss+vq_loss+0.1*attr_loss
    if is_train == "train":
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
    return  loss.item(), type_loss.item()/args.batch_size,tempo_loss.item()/args.batch_size,chord_loss.item()/args.batch_size, \
            bar_beat_loss.item()/args.batch_size, pitch_loss.item()/args.batch_size, duration_loss.item()/args.batch_size, \
            velocity_loss.item() / args.batch_size, vq_loss.item(),real_loss.item(), attr_loss.item()


def trainIter():
    max_test_loss=1000
    for epoch in range(epoch_already+1, args.Epoch):
        f = open(args.log_path, 'a')
        print("-----------------------------epoch ", epoch, "------------------------------")
        f.write('-----------------------------epoch %d------------------------------\n' % (epoch))
        train_total_loss = 0
        type_loss_total = 0;tempo_loss_total = 0;chord_loss_total = 0;bar_beat_loss_total = 0;
        pitch_loss_total = 0;duration_loss_total = 0;velocity_loss_total = 0;vq_loss_total=0;
        real_loss_total=0;attr_loss_total=0
        for i, data in enumerate(train_loader):
            input_x,input_y, length = data
            loss,type_loss,tempo_loss,chord_loss,bar_beat_loss,pitch_loss,duration_loss,velocity_loss,\
            vq_loss,real_loss,attr_loss\
            = train(input_x,input_y,length,"train")
            train_total_loss += loss
            type_loss_total+=type_loss;tempo_loss_total+=tempo_loss;chord_loss_total+=chord_loss;bar_beat_loss_total+=bar_beat_loss
            pitch_loss_total+=pitch_loss;duration_loss_total+=duration_loss;velocity_loss_total+=velocity_loss;vq_loss_total+=vq_loss
            real_loss_total+=real_loss;attr_loss_total+=attr_loss
        total_num=(i+1)
        print('epoch: %d, time: %s, \ntrain_type_loss: %.6f, train_tempo_loss: %.6f, \n'
              'train_chord_loss: %.6f, train_bar_beat_loss: %.6f,\n'
              'train_pitch_loss: %.6f, train_duration_loss: %.6f, \n'
              'train_velocity_loss: %.6f, train_vq_loss: %.10f,\n'
              'train_real_loss: %.6f, attr_loss: %.6f'
              % (epoch, timeSince(start_time), type_loss_total/total_num,tempo_loss_total/total_num,
                 chord_loss_total/total_num,bar_beat_loss_total/total_num,
                 pitch_loss_total/total_num,duration_loss_total/total_num,
                 velocity_loss_total/total_num,vq_loss_total/total_num,
                 real_loss_total/total_num,attr_loss_total / total_num))
        f.write('epoch: %d, time: %s, \ntrain_type_loss: %.6f, train_tempo_loss: %.6f, \n'
              'train_chord_loss: %.6f, train_bar_beat_loss: %.6f,\n'
              'train_pitch_loss: %.6f, train_duration_loss: %.6f, \n'
              'train_velocity_loss: %.6f, train_vq_loss: %.10f,\n'
              'train_real_loss: %.6f, attr_loss: %.6f\n'
              % (epoch, timeSince(start_time), type_loss_total/total_num,tempo_loss_total/total_num,
                 chord_loss_total/total_num,bar_beat_loss_total/total_num,
                 pitch_loss_total/total_num,duration_loss_total/total_num,
                 velocity_loss_total/total_num,vq_loss_total/total_num,
                 real_loss_total/total_num,attr_loss_total / total_num))
        loss_average=real_loss_total/total_num
        if loss_average < max_test_loss:
            print("epoch: %d save min test loss model-->test loss: %.6f" % (epoch, loss_average))
            f.write('epoch: %d save min test loss model-->test loss: %.6f\n' % (epoch, loss_average))
            model_save_path = args.model_path + f"{epoch}_best.pth"
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_save_path)
            max_test_loss = loss_average
        f.close()


if __name__=='__main__':
    setup_seed(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--Epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_width", type=int, default=128)
    parser.add_argument("--decoder_width", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=112)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--transformer_N", type=int, default=8)
    parser.add_argument("--multihead_N", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=250)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder_attention_type", type=str, default='linear')
    parser.add_argument("--decoder_attention_type", type=str, default='causal-linear')
    parser.add_argument("--cross_attention_type",  type=str, default='linear')
    parser.add_argument("--activate", type=str, default='gelu')
    parser.add_argument("--data_path", type=str, default='./data/co-representation/emopia_data.npz')
    # parser.add_argument("--data_path",  type=str, default='./data/co-representation/ailabs_data.npz')
    parser.add_argument("--dataset", type=str, default='emopia')
    parser.add_argument("--model_path", type=str, default='./saved_models/MusER_TRANS_CA_GE_emopia/')
    parser.add_argument("--load_VQ_VAE", type=str,default='')
    parser.add_argument("--log_path", type=str, default='./logs/MusER_TRANS_CA_GE_emopia.txt')
    args = parser.parse_args()

    # print params
    f = open(args.log_path, 'a')
    for arg in vars(args):
        f.write(str(arg)+"="+str(getattr(args,arg))+"\n")
    f.close()

    # preprare data
    data = np.load(args.data_path)
    train_x=data['x']
    train_y=data['y']
    data_length = data['seq_len']
    print("data_shape: ", train_x.shape, train_y.shape, data_length.shape)
    print("data_num: ", len(train_x))
    train_loader = get_data_loader(train_x, train_y, data_length)
    train_data_num=len(train_x)

    # load model
    VQ_VAE_model=VQ_VAE(args.transformer_N, args.multihead_N, args.encoder_width, args.decoder_width,
                 args.n_embeddings, args.embedding_dim, args.dropout, args.activate,
                 args.encoder_attention_type,args.decoder_attention_type,).to(device)
    model = VQ_VAE_model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss().to(device)
    epoch_already=-1
    if args.load_VQ_VAE!="":
        VQ_VAE_path=args.model_path+args.load_VQ_VAE
        VQ_VAE_dict=torch.load(VQ_VAE_path,map_location=device)
        model.load_state_dict(VQ_VAE_dict['model'])
        optimizer.load_state_dict(VQ_VAE_dict['optimizer'])
        epoch_already = VQ_VAE_dict['epoch']

    # begin training
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    start_time = time.time()
    trainIter()






