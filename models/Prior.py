import torch
import torch.nn as nn
import copy
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
from model_utils import Embeddings,PositionalEncoding
from utils import sampling

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VQ_prior(nn.Module):
    def __init__(self,N,H,codebook_n,encoder_width,dropout,activate,encoder_attention_type):
        super(VQ_prior,self).__init__()
        c=copy.deepcopy
        self.n_layer=N
        self.n_head=H
        self.encoder_size=encoder_width
        self.emb = Embeddings(codebook_n+5, 256)
        self.input_linear=nn.Linear(256,encoder_width)
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
        self.predict_fc = nn.Linear(self.encoder_size, codebook_n)
        self.softmax=nn.LogSoftmax(dim=-1)

    def forward(self,input,temperature=1):
        input_emb = self.emb(input)
        input_emb = self.input_linear(input_emb)
        input_emb = self.pos_embedd(input_emb)
        attn_mask = TriangularCausalMask(input.shape[1], device=input.device)
        h = self.transformer_encoder(input_emb, attn_mask)
        h = self.predict_fc(h)/temperature
        output = self.softmax(h)
        return output

    def inference(self, input, temperature=0.3):
        gene=torch.zeros((1, 1024)).long().to(device)
        for i in range(input.shape[1]):
            input_emb = self.emb(input)
            input_emb = self.input_linear(input_emb)
            input_emb = self.pos_embedd(input_emb)
            attn_mask = TriangularCausalMask(input.shape[1], device=input.device)
            h = self.transformer_encoder(input_emb, attn_mask)
            output = self.softmax(self.predict_fc(h)/temperature)
            output = sampling(output[0].narrow(0,i,1), p=0.9, is_training=False)
            gene[0][i]=output
            if i+1<input.shape[1]:
                input[:,i+1]=output
        return gene

