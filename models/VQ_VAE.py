import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from models.model_utils import PositionalEncoding,LayerNorm
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask
from utils import sampling
import numpy as np
from utils import gumbel_softmax

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))

    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # initialize embeddings as buffers
        embeddings = torch.empty((self.num_embeddings, self.embedding_dim))
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)

    def forward(self, x):
        flat_x=x.view(-1,self.embedding_dim)
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, H, W, C]

        # if not self.training:
        #     return quantized

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x)  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                    updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        # print("distance: ",distances.shape)
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


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
        self.emotion_emb = Embeddings(5, 128)  ###
        ### with emotion ###
        self.input_x_linear_emo = nn.Linear((32 + 128 + 256 + 64 + 512 + 128 + 128 + 128), decoder_width)
        self.input_y_linear_emo = nn.Linear((32 + 128 + 256 + 64 + 512 + 128 + 128 + 128), encoder_width)
        self.decoder_linear_emo = nn.Linear(decoder_width + embedding_dim + 128, decoder_width)
        ### encoder ###
        self.input_x_linear = nn.Linear((32 + 128 + 256 + 64 + 512 + 128 + 128), decoder_width)
        self.input_y_linear = nn.Linear((32+128+256+64+512+128+128), encoder_width)
        self.decoder_linear = nn.Linear(decoder_width+embedding_dim,decoder_width)
        self.decoder_linear_emo = nn.Linear(decoder_width + embedding_dim + 128, decoder_width)
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
        self.encoder_norm = LayerNorm(self.encoder_size)
        self.latent_linear = nn.Linear(self.encoder_size, self.embedding_dim)
        self.latent_decoder_linear = nn.Linear(self.embedding_dim, self.decoder_size)
        ### vector quantizer ###
        self.VQ=VectorQuantizerEMA(num_embeddings,embedding_dim, 0.25, 0.99, epsilon=1e-5)
        ### decoder ###
        # self.output_linear = nn.Linear(self.size, self.decoder_size)
        self.decoder_pos_embedd = nn.Sequential(c(PositionalEncoding(self.decoder_size, dropout)))
        self.transformer_decoder = TransformerDecoderBuilder.from_kwargs(
            n_layers=self.n_layer,
            n_heads=self.n_head,
            query_dimensions=self.decoder_size // self.n_head,
            value_dimensions=self.decoder_size // self.n_head,
            feed_forward_dimensions=self.decoder_size*4,
            activation=activate,
            dropout=dropout,
            attention_type=decoder_attention_type,
        ).get()
        ### predict ###
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
        tempo, chord, bar_beat, type, pitch, duration, velocity, emotion = torch.split(input, [1, 1, 1, 1, 1, 1, 1, 1],
                                                                                 dim=-1)
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

    def decoder(self, input_x, VQ_embedding):
        input_x_emb = self.input2emb(input_x)
        input_x_emb = torch.cat(input_x_emb, dim=-1)
        input_x_emb = self.input_x_linear(input_x_emb)
        # print(input_x_emb.shape,VQ_embedding.shape)
        input_x_emb= self.decoder_linear(torch.cat((input_x_emb,VQ_embedding),dim=-1))
        output = self.decoder_pos_embedd(input_x_emb)
        out_mask = TriangularCausalMask(output.shape[1], device=output.device)
        # memory_mask = FullMask(None, output.shape[1], VQ_embedding.shape[1], device=output.device)
        # print(output.shape,VQ_embedding.shape)
        # output = self.transformer_decoder(output,VQ_embedding,x_mask=out_mask,memory_mask=memory_mask)
        output = self.transformer_decoder(output, out_mask)
        return output

    def predict(self,output,GT):
        # print("predict_output: ",output.shape)
        type = self.predict_type(output)
        # topk, topi = type.topk(1)
        # type_emb = self.type_emb(topi.squeeze(-1))
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

    def prior(self,input_x,input_y):
        input = self.encoder(input_y)
        input = self.latent_linear(input)
        flat_x = input.view(-1, self.embedding_dim)
        encoding_indices = self.VQ.get_code_indices(flat_x)
        return encoding_indices

    def forward(self,input_x,input_y):
        input=self.encoder(input_x) #[b, 128, embedding_dim]
        input=self.latent_linear(input)
        quantized, loss=self.VQ(input)
        # emotion = input_x.narrow(1, 0, 1)
        # emotion_emb = self.emotion_emb(emotion.narrow(2, 7, 1))
        # emotion_emb = emotion_emb.repeat(1, self.seq_len, 1)
        # quantized = self.latent_decoder_linear(quantized)
        output = self.decoder(input_x, quantized)
        type, tempo, chord, bar_beat, pitch, duration, velocity=self.predict(output,input_y)
        type=self.softmax(type)
        tempo=self.softmax(tempo)
        chord=self.softmax(chord)
        bar_beat=self.softmax(bar_beat)
        pitch=self.softmax(pitch)
        duration=self.softmax(duration)
        velocity=self.softmax(velocity)
        return type, tempo, chord, bar_beat, pitch, duration, velocity, loss, output

    def inference(self,dictionary,memory,emotion=None):
        memory = self.latent_decoder_linear(memory)
        event2word, word2event = dictionary
        init = torch.zeros((1024,8)).long().to(device)
        init[0][-1]=emotion

        with torch.no_grad():
            final_res = []

            cnt_bar = 1
            input = init.unsqueeze(0)

            # print('------ generate ------')
            seq_i=0
            while (True):
                # sample others
                output = self.decoder(input, memory)
                type_sample, tempo, chord, bar_beat, pitch, duration, velocity=self.predict(output[0].narrow(0,seq_i,1),None)
                # print("sample_input: ",tempo.shape)
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
                # print("seq_i: ",seq_i,", next",next)
                # end of sequence
                if word2event['type'][next[3].item()] == 'EOS':
                    break

                if word2event['bar-beat'][next[2].item()] == 'Bar':
                    cnt_bar += 1

                final_res.append(np.array(next.unsqueeze(0).cpu()))
                # print(np.array(next.unsqueeze(0).cpu()).shape)

        print('--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)

        return final_res
