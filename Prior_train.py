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
import pickle
from models.MusER_TRANS_CA_GE import VQ_VAE
from models.Prior import VQ_prior
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils import clip_grad_norm_
from utils import timeSince,setup_seed

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_from_pretrained_encoder(model,data_loader):
    index=[]
    emotion=[]
    index_dict={}
    for i, prior_data in enumerate(data_loader):
        train_x, train_y=prior_data
        indices=model.prior(train_y)
        indices=indices.view(train_y.shape[0],-1)
        index.append(indices)
        emo=train_x[:,0,:].narrow(1,7,1)
        emotion.append(emo)
    index=torch.cat(index,dim=0)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            index[i][j]=VQ_dict[0][index[i][j].item()]
    padding=torch.zeros((index.shape[0],1)).to(device)
    emotion=torch.cat(emotion, dim=0)+len(VQ_dict[0])
    index_dict['x'] = torch.cat((emotion,index), dim=1)
    index_dict['y'] = torch.cat((index,padding), dim=1)
    return index_dict


def train(input_x,input_y,is_train):
    if is_train=="train":
        model.train()
    else:
        model.eval()
    output=model(input_x,temperature=0.3)
    topk,topi=output.topk(1)
    topi=topi.squeeze(-1)
    loss=0;acc=0
    for i in range(args.batch_size):
        loss += criterion(output[i], input_y[i])
        acc_temp=0
        for j in range(len(input_y[i])):
            if topi[i][j]==input_y[i][j]:
                acc_temp+=1
        acc += acc_temp / len(input_y[i])
    acc = acc/args.batch_size
    loss = loss/args.batch_size
    if is_train == "train":
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
    return loss.item(),acc


def trainIter():
    max_test_loss=10
    for epoch in range(epoch_already+1,args.Epoch):
        f = open(args.log_path, 'a')
        print("-----------------------------epoch ", epoch, "------------------------------")
        f.write('-----------------------------epoch %d------------------------------\n' % (epoch))
        total_loss = 0
        train_total_loss = 0
        total_acc=0
        for i, data in enumerate(train_loader):
            input_x,input_y = data
            loss,acc = train(input_x,input_y,"train")
            total_loss += loss
            train_total_loss += loss
            total_acc += acc
        print('epoch: %d, time: %s, train loss: %.6f, train acc: %.6f' %
              (epoch, timeSince(start_time),train_total_loss / (i+1), total_acc/(i+1)))
        f.write('epoch: %d, time: %s, train loss: %.6f, train acc: %.6f\n' %
              (epoch, timeSince(start_time),train_total_loss / (i+1), total_acc/(i+1)))
        loss_average=train_total_loss/(i+1)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--Epoch", type=int, default=1300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_width", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--transformer_N", type=int, default=8)
    parser.add_argument("--multihead_N", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=240)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder_attention_type", type=str, default='causal-linear')
    parser.add_argument("--activate", type=str, default='gelu')
    parser.add_argument("--dataset", type=str, default='emopia')
    parser.add_argument("--data_path", type=str, default='./datafile/co-representation/emopia_data.npz')
    parser.add_argument("--VQ_VAE", type=str,default='MusER_TRANS_CA_GE_emopia')
    parser.add_argument("--load_VQ_prior", type=str,default='')
    parser.add_argument("--model_path", type=str, default='./saved_models/Prior_MusER_TRANS_CA_GE_emopia/')
    parser.add_argument("--log_path", type=str, default='./logs/Prior_MusER_TRANS_CA_GE_emopia.txt')
    args = parser.parse_args()

    # print params
    f = open(args.log_path, 'a')
    f.write("-----------------------------------------\n")
    for arg in vars(args):
        f.write(str(arg)+"="+str(getattr(args,arg))+"\n")
    f.close()

    # prepare data
    VQ_dict_file = open(f'data/{args.VQ_VAE}_VQ_dict.data', 'rb')
    VQ_dict = pickle.load(VQ_dict_file)
    print(VQ_dict)
    with torch.no_grad():
        data = np.load(args.data_path)
        train_x = data['x']
        train_y = data['y']
        train_data_x = torch.LongTensor(train_x).to(device)
        train_data_y = torch.LongTensor(train_y).to(device)
        train_dataset = TensorDataset(train_data_x,train_data_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  drop_last=False)
        VQ_VAE_model=VQ_VAE(8, 8, 128, 256, 512, 112, 0.1, 'gelu', 'linear', 'causal-linear').to(device)
        if args.VQ_VAE!='':
            VQ_VAE_path=f"./saved_models/{args.VQ_VAE}/best.pt"
            model_dict=torch.load(VQ_VAE_path,map_location=device)
            VQ_VAE_model.load_state_dict(model_dict['model'])
            VQ_VAE_model=VQ_VAE_model.eval()
        data = get_from_pretrained_encoder(VQ_VAE_model, train_loader)
    train_x=data['x'].long()
    train_y=data['y'].long()
    train_data_num=train_x.shape[0]
    train_dataset = TensorDataset(train_x,train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=True)

    # load model
    model=VQ_prior(args.transformer_N, args.multihead_N, len(VQ_dict[0]), args.encoder_width,
                 args.dropout, args.activate,args.encoder_attention_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss().to(device)
    epoch_already=-1
    if args.load_VQ_Prior!='':
        VQ_prior_path={args.model_path}+{args.VQ_Prior}
        model_dict = torch.load(VQ_prior_path, map_location=device)
        model.load_state_dict(model_dict['model'])
        model = model.to(device)
        optimizer.load_state_dict(model_dict['optimizer'])
        epoch_already = model_dict['epoch']

    # begin training
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    start_time = time.time()
    trainIter()






