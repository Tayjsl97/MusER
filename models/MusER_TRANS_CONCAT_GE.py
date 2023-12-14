"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import torch
import torch.nn as nn
import copy
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask
from model_utils import Embeddings,PositionalEncoding,VectorQuantizerEMA
from utils import sampling
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VQ_VAE(nn.Module):
    def __init__(self,N,H,encoder_width,decoder_width,num_embeddings,embedding_dim,dropout,activate,
                 encoder_attention_type,decoder_attention_type):
        super(VQ_VAE,self).__init__()
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
        #  encoder
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
        self.latent_linear = nn.Linear(self.encoder_size, self.embedding_dim)
        # vector quantizer
        self.VQ=VectorQuantizerEMA(num_embeddings,embedding_dim, 0.25, 0.99, epsilon=1e-5)
        # DR
        self.VQ_pos_embedd = nn.Sequential(c(PositionalEncoding(1024, dropout)))
        self.DR = TransformerEncoderBuilder.from_kwargs(
            n_layers=4,
            n_heads=4,
            query_dimensions=1024 // 4,
            value_dimensions=1024 // 4,
            feed_forward_dimensions=1024 * 4,
            activation=activate,
            dropout=dropout,
            attention_type=encoder_attention_type,
        ).get()
        # decoder
        self.input_x_linear_emo = nn.Linear((32 + 128 + 256 + 64 + 512 + 128 + 128 + 128), decoder_width)
        self.decoder_linear = nn.Linear(decoder_width + embedding_dim, decoder_width)
        self.decoder_pos_embedd = nn.Sequential(c(PositionalEncoding(self.decoder_size, dropout)))
        self.transformer_decoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=4,
            n_heads=self.n_head,
            query_dimensions=self.decoder_size // self.n_head,
            value_dimensions=self.decoder_size // self.n_head,
            feed_forward_dimensions=self.decoder_size*4,
            activation=activate,
            dropout=dropout,
            attention_type=decoder_attention_type,
        ).get()
        self.subDecoder_linear = nn.Linear(16, decoder_width)
        self.type_decoder=TransformerEncoderBuilder.from_kwargs(
            n_layers=2,
            n_heads=self.n_head,
            query_dimensions=self.decoder_size // self.n_head,
            value_dimensions=self.decoder_size // self.n_head,
            feed_forward_dimensions=self.decoder_size*4,
            activation=activate,
            dropout=dropout,
            attention_type=decoder_attention_type,
        ).get()
        self.tempo_decoder = copy.deepcopy(self.type_decoder)
        self.chord_decoder = copy.deepcopy(self.type_decoder)
        self.beat_decoder = copy.deepcopy(self.type_decoder)
        self.pitch_decoder = copy.deepcopy(self.type_decoder)
        self.duration_decoder = copy.deepcopy(self.type_decoder)
        self.velocity_decoder = copy.deepcopy(self.type_decoder)
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
        tempo, chord, bar_beat, type, pitch, duration, velocity, emotion\
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
        input_y_emb_origin = self.input2emb(input_y)
        input_y_emb = torch.cat(input_y_emb_origin[:-1], dim=-1)
        input_y_emb = self.input_y_linear(input_y_emb)
        input = self.pos_embedd(input_y_emb)
        attn_mask = FullMask(None, input.shape[1],input.shape[1],device=input.device)
        h = self.transformer_encoder(input, attn_mask)
        return h

    def decoder(self, input_x, VQ_embedding):
        input_x_emb = self.input2emb(input_x)
        input_x_emb = torch.cat(input_x_emb, dim=-1)
        input_x_emb = self.input_x_linear_emo(input_x_emb)
        input_x_emb = self.decoder_linear(torch.cat((input_x_emb, VQ_embedding), dim=-1))  # concat
        output = self.decoder_pos_embedd(input_x_emb)
        out_mask = TriangularCausalMask(output.shape[1], device=output.device)
        output = self.transformer_decoder(output,out_mask)
        return output

    def subDecoder(self,decoder,predict_fc,output,ele,type_sample,flag):
        output=output+self.subDecoder_linear(ele)
        output=self.decoder_pos_embedd(output)
        out_mask = TriangularCausalMask(output.shape[1], device=output.device)
        output=decoder(output,out_mask)
        if flag=="type":
            output=predict_fc(output)
        else:
            type_emb = self.type_emb(type_sample[:, :, 3])  # training
            # type_emb = self.type_emb(type_sample) # inference
            output=self.project_concat_type(torch.cat((output,type_emb),dim=-1))
            output=predict_fc(output)
        return output

    def prior(self,input_y):
        input = self.encoder(input_y)
        input = self.latent_linear(input)
        flat_x = input.view(-1, self.embedding_dim)
        encoding_indices = self.VQ.get_code_indices(flat_x)
        return encoding_indices

    def compute_reg_loss(self,input,ele_laten_var):
        input_ele = torch.split(input, [1, 1, 1, 1, 1, 1, 1, 1], dim=-1)
        loss=0
        spread_weight=1
        for i in range(7):
            input_temp=input_ele[i].squeeze(-1)
            ele_temp=ele_laten_var[i]
            input_diff_matrix=torch.zeros((input.shape[0],input.shape[0],ele_temp.shape[-1])).to(device)
            ele_diff_matrix=torch.zeros((input.shape[0],input.shape[0],ele_temp.shape[-1])).to(device)
            for j in range(input.shape[0]):
                for k in range(input.shape[0]):
                    input_diff_matrix[j][k]=input_temp[j]-input_temp[k]
                    ele_diff_matrix[j][k]=ele_temp[j]-ele_temp[k]
            loss_temp=nn.L1Loss(reduction='mean')(torch.tanh(spread_weight*ele_diff_matrix),torch.sign(input_diff_matrix))
            loss+=loss_temp
        return loss

    def get_latent_var(self,quantized):
        latent_var_list=[]
        for i in range(7):
            quantized_temp=quantized[:,:,16*i:(i+1)*16].transpose(1,2)
            quantized_temp=self.VQ_pos_embedd(quantized_temp)
            quantized_temp=self.DR(quantized_temp)[:,0,:]
            latent_var_list.append(quantized_temp)
        return latent_var_list

    def forward(self,input_x,input_y):
        # encode
        input=self.encoder(input_y)
        input=self.latent_linear(input)
        # VQ
        quantized, loss=self.VQ(input)
        # regularization
        ele_latent_var = self.get_latent_var(quantized)
        reg_loss=self.compute_reg_loss(input_y,ele_latent_var)
        # decode
        output = self.decoder(input_x, quantized)
        type = self.subDecoder(self.type_decoder, self.predict_type, output, quantized[:, :, 48:64], input_y, "type")
        tempo = self.subDecoder(self.tempo_decoder, self.predict_tempo, output, quantized[:, :, 0:16], input_y,"nontype")
        chord = self.subDecoder(self.chord_decoder, self.predict_chord, output, quantized[:, :, 16:32], input_y,"nontype")
        beat = self.subDecoder(self.beat_decoder, self.predict_bar_beat, output, quantized[:, :, 32:48], input_y,"nontype")
        pitch = self.subDecoder(self.pitch_decoder, self.predict_pitch, output, quantized[:, :, 64:80], input_y,"nontype")
        duration = self.subDecoder(self.duration_decoder, self.predict_duration, output, quantized[:, :, 80:96],input_y, "nontype")
        velocity = self.subDecoder(self.velocity_decoder, self.predict_velocity, output, quantized[:, :, 96:112],input_y, "nontype")
        type=self.softmax(type)
        tempo=self.softmax(tempo)
        chord=self.softmax(chord)
        beat=self.softmax(beat)
        pitch=self.softmax(pitch)
        duration=self.softmax(duration)
        velocity=self.softmax(velocity)
        return type, tempo, chord, beat, pitch, duration, velocity, loss, reg_loss

    def inference(self,dictionary,quantized,emotion=None):
        event2word, word2event = dictionary
        init = torch.zeros((1024,8)).long().to(device)
        init[0][-1]=emotion;init[0][3]=1
        type_sampling_list=torch.zeros((1,1024)).long().to(device)
        with torch.no_grad():
            final_res = []

            cnt_bar = 0
            input = init.unsqueeze(0)
            quantized_type = quantized[:, :, 48:64]
            quantized_tempo = quantized[:, :, 0:16]
            quantized_chord = quantized[:, :, 16:32]
            quantized_beat = quantized[:, :, 32:48]
            quantized_pitch = quantized[:, :, 64:80]
            quantized_duration = quantized[:, :, 80:96]
            quantized_velocity = quantized[:, :, 96:112]
            seq_i=0
            while (True):
                output = self.decoder(input, quantized)
                type = self.subDecoder(self.type_decoder, self.predict_type, output, quantized_type, None, "type")
                type_sample = sampling(type[:,seq_i,:], p=0.9, is_training=False)
                type_sampling_list[:,seq_i]=type_sample
                tempo = self.subDecoder(self.tempo_decoder, self.predict_tempo, output, quantized_tempo, type_sampling_list,"nontype")
                chord = self.subDecoder(self.chord_decoder, self.predict_chord, output, quantized_chord, type_sampling_list,"nontype")
                beat = self.subDecoder(self.beat_decoder, self.predict_bar_beat, output, quantized_beat, type_sampling_list,"nontype")
                pitch = self.subDecoder(self.pitch_decoder, self.predict_pitch, output, quantized_pitch, type_sampling_list,"nontype")
                duration = self.subDecoder(self.duration_decoder, self.predict_duration, output, quantized_duration,type_sampling_list, "nontype")
                velocity = self.subDecoder(self.velocity_decoder, self.predict_velocity, output, quantized_velocity,type_sampling_list, "nontype")
                tempo_sample = sampling(tempo[:,seq_i,:], t=1.2, p=0.9)
                barbeat_sample = sampling(beat[:,seq_i,:], t=1.2)
                chord_sample = sampling(chord[:,seq_i,:], p=0.99)
                pitch_sample = sampling(pitch[:,seq_i,:], p=0.9)
                duration_sample = sampling(duration[:,seq_i,:], t=2, p=0.9)
                velocity_sample = sampling(velocity[:,seq_i,:], t=5)

                next = torch.LongTensor([
                    tempo_sample,
                    chord_sample,
                    barbeat_sample,
                    type_sample,
                    pitch_sample,
                    duration_sample,
                    velocity_sample,
                    0
                ]).to(device)
                input[0, seq_i + 1, :] = next
                seq_i += 1
                if seq_i == 1023:
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
