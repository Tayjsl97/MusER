"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import numpy as np
import argparse
import torch
import os
import pickle
from models.MusER_TRANS_CA_GE import VQ_VAE
from torch.utils.data import DataLoader,TensorDataset
from utils import write_midi

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_from_pretrained_encoder(model,data_loader):
    index=[]
    for i, prior_data in enumerate(data_loader):
        train_x, train_y=prior_data
        indices = model.prior(train_y)
        indices=indices.view(train_y.shape[0], -1)
        index.append(indices)
    index=torch.cat(index,dim=0)
    feature = model.VQ.quantize(index)
    return index, feature


def generate_from_prior(latent,music_name):
    path_dictionary = "./data/co-representation/dictionary.pkl"
    with open(path_dictionary, "rb") as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary
    res = VQ_VAE_model.inference(dictionary,latent,emotion=emotion_tag)
    music_name = "emotion"+str(emotion_tag)+"_"+str(music_name)
    midi_path = os.path.join(args.music_path,music_name+".mid")
    write_midi(res, str(midi_path), word2event)
    return res


def generate_music(res,i):
    path_dictionary = "./data/co-representation/dictionary.pkl"
    if not os.path.exists(args.music_path):
        os.makedirs(args.music_path)
    with open(path_dictionary, "rb") as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary
    midi_path = os.path.join(args.music_path, "original_emotion"+str(emotion_tag)+"_"+str(i) + ".mid")
    write_midi(res, str(midi_path), word2event)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/co-representation/emopia_data.npz')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--VQ_VAE", type=str, default='')
    parser.add_argument("--music_path", type=str, default='./transfer_midi/4-1v/')
    args = parser.parse_args()
    if not os.path.exists(args.music_path):
        os.makedirs(args.music_path)

    data = np.load('./data/co-representation/emopia_idx.npz')
    ax1_index = data['cls_1_idx']
    ax2_index = data['cls_2_idx']
    ax3_index = data['cls_3_idx']
    ax4_index = data['cls_4_idx']
    data = np.load(args.data_path)
    train_x = data['x']
    train_y = data['y']
    data_length = data['seq_len']
    train_data_x = torch.LongTensor(train_x).to(device)
    train_data_y = torch.LongTensor(train_y).to(device)
    train_dataset = TensorDataset(train_data_x, train_data_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    number = 20
    with torch.no_grad():
        model = VQ_VAE(8, 8, 128, 256, 512, 112, 0.1, 'gelu', 'linear', 'causal-linear').to(device)
        if args.VQ_VAE!="":
            VQ_VAE_path = f"./saved_models/{args.VQ_VAE}/best.pt"
            model_dict = torch.load(VQ_VAE_path ,map_location=device)
            model.load_state_dict(model_dict['model'])
            VQ_VAE_model =model.to(device).eval()
        data, VQ_feature = get_from_pretrained_encoder(VQ_VAE_model,train_loader)
        ax1_list=[];ax4_list=[]
        cnt = 0
        for i in ax1_index:
            ax1_list.append(VQ_feature[i])
            cnt+=1
            if cnt == number:
                break
        ax1_list=torch.stack(ax1_list)
        cnt = 0
        for i in ax4_index:
            ax4_list.append(VQ_feature[i])
            cnt += 1
            if cnt == number:
                break
        ax4_list=torch.stack(ax4_list)
        change1 = ax4_list[:, :, :96]
        change2 = ax1_list[:, :, 96:112]
        latent=torch.cat((change1,change2),dim=-1)
        data=[]
        index = {"cls_1_idx": [], "cls_2_idx": [], "cls_3_idx": [], "cls_4_idx": []}
        emotion_tag=1
        cnt=0
        for i in ax4_index:
            data.append(train_y[i])
            index["cls_4_idx"].append(cnt)
            generate_music(train_y[i],cnt)
            cnt+=1
            if cnt==number:
                break
        for i in range(latent.shape[0]):
            res = generate_from_prior(latent.narrow(0,i,1), i)
            index["cls_1_idx"].append(cnt)
            data.append(res)
            cnt+=1

        index_file = open("transfer_midi/4-1v_index.data", 'wb')
        pickle._dump(index, index_file)
        index_file.close()
        data_file = open("transfer_midi/4-1v_data.data", 'wb')
        pickle._dump(data, data_file)
        data_file.close()


