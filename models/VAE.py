"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import torch
import torch.nn as nn
import copy
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask
from model_utils import Embeddings,PositionalEncoding
from utils import sampling
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self,N,H,encoder_width,decoder_width,embedding_dim,dropout,activate,
                 encoder_attention_type,decoder_attention_type):
        super(VAE,self).__init__()
        c=copy.deepcopy
        self.n_layer=N
        self.n_head=H
        self.encoder_size=encoder_width
        self.embedding_dim=embedding_dim
        self.decoder_size = decoder_width
        self.type_emb = Embeddings(4, 32)
        self.tempo_emb = Embeddings(56, 128)
        self.chord_emb = Embeddings(135, 256)
        self.bar_beat_emb = Embeddings(18, 64)
        self.pitch_emb = Embeddings(87, 512)
        self.duration_emb = Embeddings(18, 128)
        self.velocity_emb = Embeddings(42, 128)
        self.emotion_emb = Embeddings(5, 128)
        # encoder
        self.input_y_linear = nn.Linear((32 + 128 + 256 + 64 + 512 + 128 + 128), encoder_width)
        self.pos_embedd = nn.Sequential(c(PositionalEncoding(self.encoder_size, dropout)))
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layer,
            n_heads=self.n_head,
            query_dimensions=self.encoder_size // self.n_head,
            value_dimensions=self.encoder_size // self.n_head,
            feed_forward_dimensions=self.encoder_size*4,
            activation=activate,
            dropout=dropout,
            attention_type=encoder_attention_type,
        ).get()
        # z
        self.latent_linear = nn.Linear(self.encoder_size, self.embedding_dim)
        self.latent_emo_linear=nn.Linear(self.embedding_dim+128,self.embedding_dim)
        self.mean_linear=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.logv_linear=nn.Linear(self.embedding_dim,self.embedding_dim)
        # decoder
        self.input_x_linear = nn.Linear((32 + 128 + 256 + 64 + 512 + 128 + 128), decoder_width)
        self.decoder_linear = nn.Linear(decoder_width + embedding_dim, decoder_width)
        self.decoder_pos_embedd = nn.Sequential(c(PositionalEncoding(self.decoder_size, dropout)))
        self.transformer_decoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layer,
            n_heads=self.n_head,
            query_dimensions=self.decoder_size // self.n_head,
            value_dimensions=self.decoder_size // self.n_head,
            feed_forward_dimensions=self.decoder_size*4,
            activation=activate,
            dropout=dropout,
            attention_type=decoder_attention_type,
        ).get()
        # predict
        self.predict_type=nn.Linear(self.decoder_size, 4)
        self.project_concat_type = nn.Linear(self.decoder_size + 32, self.decoder_size)
        self.predict_tempo = nn.Linear(self.decoder_size, 56)
        self.predict_chord = nn.Linear(self.decoder_size, 135)
        self.predict_bar_beat = nn.Linear(self.decoder_size, 18)
        self.predict_pitch = nn.Linear(self.decoder_size, 87)
        self.predict_duration = nn.Linear(self.decoder_size, 18)
        self.predict_velocity = nn.Linear(self.decoder_size, 42)
        self.softmax=nn.LogSoftmax(dim=-1)

    def input2emb(self,input):
        tempo, chord, bar_beat, type, pitch, duration, velocity, emotion \
            = torch.split(input, [1, 1, 1, 1, 1, 1, 1, 1], dim=-1)
        type_emb = self.type_emb(type.squeeze(-1))
        tempo_emb = self.tempo_emb(tempo.squeeze(-1))
        chord_emb = self.chord_emb(chord.squeeze(-1))
        bar_beat_emb = self.bar_beat_emb(bar_beat.squeeze(-1))
        pitch_emb = self.pitch_emb(pitch.squeeze(-1))
        duration_emb = self.duration_emb(duration.squeeze(-1))
        velocity_emb = self.velocity_emb(velocity.squeeze(-1))
        emotion_emb = self.emotion_emb(emotion.squeeze(-1))
        return (type_emb,tempo_emb,chord_emb,bar_beat_emb,pitch_emb,duration_emb,velocity_emb,emotion_emb)


    def encoder(self,input_y):
        input_y_emb = self.input2emb(input_y)
        input_y_emb = torch.cat(input_y_emb[:-1], dim=-1)
        input_y_emb = self.input_y_linear(input_y_emb)
        input = self.pos_embedd(input_y_emb)
        attn_mask = FullMask(None, input.shape[1],input.shape[1],device=input.device)
        h = self.transformer_encoder(input, attn_mask)
        return h

    def decoder(self, input_x, z_embedding):
        input_x_emb = self.input2emb(input_x)
        input_x_emb = torch.cat(input_x_emb[:-1], dim=-1)
        input_x_emb = self.input_x_linear(input_x_emb)
        z_embedding=z_embedding.unsqueeze(1).repeat(1,1024,1)
        input_x_emb= self.decoder_linear(torch.cat((input_x_emb,z_embedding),dim=-1))
        output = self.decoder_pos_embedd(input_x_emb)
        out_mask = TriangularCausalMask(output.shape[1], device=output.device)
        output = self.transformer_decoder(output, out_mask)
        return output

    def predict(self,output,GT):
        type = self.predict_type(output)
        if GT!=None:
            type_emb=self.type_emb(GT[:,:,3])
        else:
            type_sample=sampling(self.predict_type(output),p=0.9,is_training=False)
            type_emb = self.type_emb(torch.LongTensor([type_sample]).to(device))
        output = self.project_concat_type(torch.cat((output, type_emb), dim=-1))
        tempo = self.predict_tempo(output)
        chord = self.predict_chord(output)
        bar_beat = self.predict_bar_beat(output)
        pitch = self.predict_pitch(output)
        duration = self.predict_duration(output)
        velocity = self.predict_velocity(output)
        if GT!=None:
            return type,tempo,chord,bar_beat,pitch,duration,velocity
        else:
            return type_sample, tempo, chord, bar_beat, pitch, duration, velocity

    def prior(self,input_y):
        input = self.encoder(input_y)
        input = self.latent_linear(input)
        latent = input.mean(dim=1)
        mean = self.mean_linear(latent)
        logv = self.logv_linear(latent)
        std = torch.exp(0.5 * logv)
        z = torch.randn([mean.shape[0], self.embedding_dim]).to(device)
        z = z * std + mean
        return z

    def forward(self,input_x,input_y):
        input=self.encoder(input_y)
        input=self.latent_linear(input)
        latent=input.mean(dim=1)
        mean=self.mean_linear(latent)
        logv=self.logv_linear(latent)
        std=torch.exp(0.5*logv)
        z=torch.randn([mean.shape[0], self.embedding_dim]).to(device)
        z=z*std+mean
        emo=self.emotion_emb(input_x[:,0,-1])
        z=self.latent_emo_linear(torch.cat((z,emo),dim=-1))
        kl_loss=-0.5*torch.sum(1+logv-mean.pow(2)-logv.exp())
        output = self.decoder(input_x, z)
        type, tempo, chord, bar_beat, pitch, duration, velocity=self.predict(output,input_y)
        type=self.softmax(type)
        tempo=self.softmax(tempo)
        chord=self.softmax(chord)
        bar_beat=self.softmax(bar_beat)
        pitch=self.softmax(pitch)
        duration=self.softmax(duration)
        velocity=self.softmax(velocity)
        return type, tempo, chord, bar_beat, pitch, duration, velocity, kl_loss

    def inference(self,dictionary,memory,emotion=None):
        event2word, word2event = dictionary
        init = torch.zeros((1024,8)).long().to(device)
        init[0][-1]=emotion;init[0][3]=1

        with torch.no_grad():
            final_res = []

            cnt_bar = 1
            input = init.unsqueeze(0)

            seq_i=0
            while (True):
                # sample others
                output = self.decoder(input, memory)
                type_sample, tempo, chord, bar_beat, pitch, duration, velocity=self.predict(output[0].narrow(0,seq_i,1),None)
                tempo_sample = sampling(tempo, t=1.2, p=0.9)
                barbeat_sample = sampling(bar_beat, t=1.2)
                chord_sample = sampling(chord, p=0.99)
                pitch_sample = sampling(pitch, p=0.9)
                duration_sample = sampling(duration, t=2, p=0.9)
                velocity_sample = sampling(velocity, t=5)
                next=torch.LongTensor([
                    tempo_sample,
                    chord_sample,
                    barbeat_sample,
                    type_sample,
                    pitch_sample,
                    duration_sample,
                    velocity_sample,
                    0
                ]).to(device)
                input[0, seq_i+1, :]=next
                seq_i+=1
                if seq_i==1023:
                    break

                if word2event['type'][next[3].item()] == 'EOS':
                    break

                if word2event['bar-beat'][next[2].item()] == 'Bar':
                    cnt_bar += 1

                final_res.append(np.array(next.unsqueeze(0).cpu()))

        print('--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)

        return final_res
