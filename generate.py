"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import os
import pickle
import time
import numpy as np
from utils import write_midi_new
import argparse
import torch
from models.MusER_TRANS_CA_GE import VQ_VAE
from models.Prior import VQ_prior

np.set_printoptions(threshold=np.inf)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(latent,music_name):
    path_dictionary = "./data/co-representation/dictionary.pkl"
    with open(path_dictionary, "rb") as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary
    res = VQ_VAE_model.inference(dictionary,latent,emotion=emotion_tag)
    music_name="emotion"+str(emotion_tag)+"_"+str(music_name)
    midi_path=os.path.join(music_path,music_name+".mid")
    write_midi_new(res, str(midi_path), word2event)
    return res


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--VQ_prior", type=str, default='Prior_TRANS_CA_ER_emopia')
    parser.add_argument("--VQ_VAE", type=str, default='MusER_TRANS_CA_ER_emopia')
    parser.add_argument("--music_file", type=str, default='MusER_TRANS_CA_ER_emopia')
    args = parser.parse_args()
    music_path=f'./generate_midi/{args.music_file}/'
    if not os.path.exists(music_path):
        os.makedirs(music_path)

    # the embedding index in the codebook actually used in model training
    VQ_dict_file = open('data/VQ_dict.data', 'rb')
    VQ_dict = pickle.load(VQ_dict_file)
    # print(VQ_dict[0])
    # print(VQ_dict[1])
    ####################
    VQ_VAE_path=f'./saved_models/{args.VQ_VAE}/best.pt'
    VQ_VAE_model=VQ_VAE(8, 8, 128, 256, 512, 112, 0.1, 'gelu', 'linear', 'causal-linear').to(device)
    VQ_VAE_dict = torch.load(VQ_VAE_path, map_location=device)
    VQ_VAE_model.load_state_dict(VQ_VAE_dict['model'])
    VQ_VAE_model.eval()
    #####################
    VQ_prior_path=f'./saved_models/{args.VQ_prior}/best.pt'
    codebook_n=len(VQ_dict[0])
    VQ_prior_model = VQ_prior(8, 8, codebook_n, 256, 0.1, 'gelu','causal-linear').to(device)
    dict = torch.load(VQ_prior_path,map_location=device)
    VQ_prior_model.load_state_dict(dict['model'])
    VQ_prior_model = VQ_prior_model.to(device).eval()
    ######################
    index = {"cls_1_idx": [], "cls_2_idx": [], "cls_3_idx": [], "cls_4_idx": []}
    data = []
    start_time=time.time()
    with torch.no_grad():
        for j in range(1,5):
            emotion_tag = j
            for i in range(100):
                print("--------"+str(j)+"__"+ str(i) + "--------")
                init = torch.zeros((1, 1024)).long().to(device)
                init[0][0]=emotion_tag+codebook_n
                indices=VQ_prior_model.inference(init)
                for k in range(indices.shape[1]):
                    indices[0][k]=VQ_dict[1][indices[0][k].item()]
                latent = VQ_VAE_model.VQ.quantize(indices)
                res=generate(latent,i)
                index_name="cls_"+str(j)+"_idx"
                index[index_name].append(i+100*(j-1))
                data.append(res)
    index_file=open(f"generate_midi/{args.VQ_VAE}_index.data",'wb')
    pickle._dump(index,index_file)
    index_file.close()
    data_file=open(f"generate_midi/{args.VQ_VAE}_data.data",'wb')
    pickle._dump(data,data_file)
    data_file.close()