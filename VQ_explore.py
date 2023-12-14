"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import numpy as np
import argparse
import torch
from models.MusER_TRANS_CA_GE import VQ_VAE
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.metrics import silhouette_score
import pandas as pd
import pickle

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


def scatter_plot(x,y,Y):
    sns.scatterplot(
        x=x,
        y=y,
        hue=Y,
        style=Y,
        palette="Dark2",
        legend="full",
        s=100,
    )
    font1 = {'weight': 'normal', 'size': 14}
    legend = plt.legend(title="Quadrant", loc="lower right", fancybox=True, prop=font1)
    plt.setp(legend.get_title(), fontsize=14)
    plt.xticks([])
    plt.yticks([])


def t_sne(feature, input_x, name, plot_flag):
    Y=[]
    new_feature=torch.zeros_like(feature)
    j=0
    for i in range(feature.shape[0]):
        if input_x[i,0,-1].item()==1:
            Y.append("Q1")
            new_feature[j]=feature[i]
            j+=1
    print(j)
    for i in range(feature.shape[0]):
        if input_x[i,0,-1].item()==2:
            Y.append("Q2")
            new_feature[j] = feature[i]
            j += 1
    print(j)
    for i in range(feature.shape[0]):
        if input_x[i,0,-1].item()==3:
            Y.append("Q3")
            new_feature[j] = feature[i]
            j += 1
    print(j)
    for i in range(feature.shape[0]):
        if input_x[i, 0, -1].item() == 4:
            Y.append("Q4")
            new_feature[j] = feature[i]
            j += 1
    print(j)
    new_feature=new_feature[:len(Y)]
    tsne = manifold.TSNE(n_components=2, perplexity=30, init='pca', n_iter=3000, random_state=13, learning_rate='auto')
    tsne=tsne.fit_transform(new_feature.mean(dim=1).squeeze(1))
    tsne_df=pd.DataFrame(tsne)
    plt.figure(figsize=(12, 7.5))
    htw=tsne_df.to_numpy(copy=True)
    ################ compute Silhouette Coefficient ################
    # Q1-Q2
    X_temp=np.concatenate((htw[:239, :], htw[239:501, :]), axis=0)
    Label_temp1=np.array([0]*239)
    Label_temp2=np.array([1]*262)
    Label_temp=np.concatenate((Label_temp1,Label_temp2),axis=0)
    Q1_Q2_SC=silhouette_score(X_temp,Label_temp)
    # Q1-Q3
    X_temp = np.concatenate((htw[:239, :], htw[501:745, :]), axis=0)
    Label_temp1 = np.array([0] * 239)
    Label_temp2 = np.array([1] * 244)
    Label_temp = np.concatenate((Label_temp1, Label_temp2), axis=0)
    Q1_Q3_SC = silhouette_score(X_temp, Label_temp)
    # Q1-Q4
    X_temp = np.concatenate((htw[:239, :], htw[745:1052, :]), axis=0)
    Label_temp1 = np.array([0] * 239)
    Label_temp2 = np.array([1] * 307)
    Label_temp = np.concatenate((Label_temp1, Label_temp2), axis=0)
    Q1_Q4_SC = silhouette_score(X_temp, Label_temp)
    # Q3-Q4
    X_temp = np.concatenate((htw[501:745, :], htw[745:1052, :]), axis=0)
    Label_temp1 = np.array([0] * 244)
    Label_temp2 = np.array([1] * 307)
    Label_temp = np.concatenate((Label_temp1, Label_temp2), axis=0)
    Q3_Q4_SC = silhouette_score(X_temp, Label_temp)
    # Q2-Q3
    X_temp = np.concatenate((htw[239:501, :], htw[501:745, :]), axis=0)
    Label_temp1 = np.array([0] * 262)
    Label_temp2 = np.array([1] * 244)
    Label_temp = np.concatenate((Label_temp1, Label_temp2), axis=0)
    Q2_Q3_SC = silhouette_score(X_temp, Label_temp)
    # Q2-Q4
    X_temp = np.concatenate((htw[239:501, :], htw[745:1052, :]), axis=0)
    Label_temp1 = np.array([0] * 262)
    Label_temp2 = np.array([1] * 307)
    Label_temp = np.concatenate((Label_temp1, Label_temp2), axis=0)
    Q2_Q4_SC = silhouette_score(X_temp, Label_temp)
    print("Silhouette Coefficient: ")
    print("Q1_Q2: ", round(Q1_Q2_SC,4), ", Q1_Q3: ", round(Q1_Q3_SC,4), ", Q1_Q4: ", round(Q1_Q4_SC,4),
          "Q3_Q4: ", round(Q3_Q4_SC,4), ", Q2_Q3: ", round(Q2_Q3_SC,4), ", Q2_Q4: ", round(Q2_Q4_SC,4))
    if plot_flag==True:
        plt.subplot(2, 3, 1)
        x = np.concatenate((htw[:239, 0], htw[239:501, 0]), axis=0)
        y = np.concatenate((htw[:239, 1], htw[239:501, 1]), axis=0)
        new_Y = Y[:239];
        new_Y.extend(Y[239:501])
        scatter_plot(x,y,new_Y,name)
        # Q1-Q3
        plt.subplot(2, 3, 3)
        x = np.concatenate((htw[:239, 0], htw[501:745, 0]), axis=0)
        y = np.concatenate((htw[:239, 1], htw[501:745, 1]), axis=0)
        new_Y = Y[:239];
        new_Y.extend(Y[501:745])
        scatter_plot(x, y, new_Y,name)
        # Q1-Q4
        plt.subplot(2, 3, 2)
        x = np.concatenate((htw[:239, 0], htw[745:1052, 0]), axis=0)
        y = np.concatenate((htw[:239, 1], htw[745:1052, 1]), axis=0)
        new_Y = Y[:239];
        new_Y.extend(Y[745:1052])
        scatter_plot(x, y, new_Y,name)
        # Q3-Q4
        plt.subplot(2, 3, 4)
        x = np.concatenate((htw[501:745, 0], htw[745:1052, 0]), axis=0)
        y = np.concatenate((htw[501:745, 1], htw[745:1052, 1]), axis=0)
        new_Y = Y[501:745];
        new_Y.extend(Y[745:1052])
        scatter_plot(x, y, new_Y,name)
        # Q2-Q3
        plt.subplot(2, 3, 5)
        x = np.concatenate((htw[239:501, 0],htw[501:745, 0]), axis=0)
        y = np.concatenate((htw[239:501, 1],htw[501:745, 1]), axis=0)
        new_Y = Y[239:501];
        new_Y.extend(Y[501:745])
        scatter_plot(x, y, new_Y,name)
        # Q2-Q4
        plt.subplot(2, 3, 6)
        x = np.concatenate((htw[239:501, 0], htw[745:1052, 0]), axis=0)
        y = np.concatenate((htw[239:501, 1], htw[745:1052, 1]), axis=0)
        new_Y = Y[239:501];
        new_Y.extend(Y[745:1052])
        scatter_plot(x, y, new_Y,name)
        plt.tight_layout()
        eps = plt.gcf()
        eps.savefig('img/'+name+'.png', dpi=600, format='png', bbox_inches='tight')
    return new_feature


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default='./data/co-representation/emopia_data.npz')
    parser.add_argument("--VQ_VAE", type=str,default='')
    parser.add_argument("--plot_flag", type=bool, default=True)
    args = parser.parse_args()

    with torch.no_grad():
        data = np.load(args.data_path)
        train_x = data['x']
        train_y = data['y']
        data_length = data['seq_len']
        train_data_x = torch.LongTensor(train_x).to(device)
        train_data_y = torch.LongTensor(train_y).to(device)
        train_dataset = TensorDataset(train_data_x ,train_data_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        model =VQ_VAE(8, 8, 128, 256, 512, 112, 0.1, 'gelu', 'linear', 'causal-linear').to(device)
        if args.VQ_VAE!="":
            VQ_VAE_path = f"./saved_models/{args.VQ_VAE}/best.pt"
            model_dict = torch.load(VQ_VAE_path ,map_location=device)
            model.load_state_dict(model_dict['model'])
            VQ_VAE_model =model.to(device).eval()
        indices, VQ_feature = get_from_pretrained_encoder(VQ_VAE_model,train_loader)
        ##############################################
        # saving VQ dict
        indices = indices.cpu().numpy()
        class_n=set()
        for i in range(len(indices)):
            class_n=class_n|set(indices[i])
        class_dict={0:{},1:{}}
        for j in class_n:
            class_dict[0][j]=len(class_dict[0])
            class_dict[1][len(class_dict[1])]=j
        file=open(f'data/{args.VQ_VAE}_VQ_dict.data', 'wb')
        pickle._dump(class_dict,file)
        file.close()
        ################################################
        name=["Tempo","Chord","Beat","Family","Pitch","Duration","Velocity"]
        for i in range(7):
            print("-------------"+name[i]+"----------------")
            t_sne(VQ_feature[:,:,i*16:(i+1)*16].cpu(), train_x, args.VQ_VAE + "_"+name[i], args.plot_flag)
        print("-------------all----------------")
        t_sne(VQ_feature.cpu(), train_x, args.VQ_VAE + "_all", args.plot_flag)
