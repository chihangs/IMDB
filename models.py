import math
import numpy as np
import os, sys, re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


#adapted from: https://www.kaggle.com/code/miladlink/imdb-sentiment-analysis-pytorch

# Attention computes re-weighted hidden states of the LSTM Model
# In fact, it produces a weight for each hidden state at different time steps

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    def forward(self, encoder_outputs):
        # encoder_outputs: [batch size, sent len, hid dim]
        energy = self.projection(encoder_outputs)
        # energy: [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights: [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # outputs: [batch size, hid dim]
        return outputs, weights

class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout= 0 if n_layers == 1 else dropout)
        self.attention = SelfAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [sent len, batch size]
        embedded = self.embedding(x)
        # embedded: [sent len, batch size, emb dim]
        output, (hidden, cell) = self.lstm(embedded)
        # use 'batch_first' if you want batch size to be the 1st para
        # output: [sent len, batch size, hid dim*num directions]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]    #align bi-directional to single direction shape
        # output: [sent len, batch size, hid dim]
        ouput = output.permute(1, 0, 2)
        # ouput: [batch size, sent len, hid dim]
        new_embed, weights = self.attention(ouput)
        # new_embed: [batch size, hid dim], weights: [batch size, sent len]
        new_embed = self.dropout(new_embed)
        new_embed = self.fc(new_embed).squeeze()
        return new_embed

#-----------------------------------------------------------------------------------------------------------------
class LSTM(nn.Module):                         #modified from above for ablation study
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, MAX_LEN, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout= 0 if n_layers == 1 else dropout)
        #self.attention = SelfAttention(hidden_dim)           #****************cut this line
        self.flat = nn.Flatten()
        self.fc = nn.Linear(hidden_dim * MAX_LEN, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [sent len, batch size]
        embedded = self.embedding(x)
        # embedded: [sent len, batch size, emb dim]
        output, (hidden, cell) = self.lstm(embedded)
        # use 'batch_first' if you want batch size to be the 1st para
        # output: [sent len, batch size, hid dim*num directions]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output: [sent len, batch size, hid dim]
        output = output.permute(1, 0, 2)
        # ouput: [batch size, sent len, hid dim]
        #new_embed, weights = self.attention(ouput)     #****************cut this line
        # new_embed: [batch size, hid dim]
        # weights: [batch size, sent len]
        output = self.flat(output)
        new_embed = self.dropout(output)   #*************change argument to output
        new_embed = self.fc(new_embed).squeeze()         #****************** squeeze to adjust size
        return new_embed
#-----------------------------------------------------------------------------------------------------------------
class GRU(nn.Module):                         #modified from above for ablation study
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, MAX_LEN, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout= 0 if n_layers == 1 else dropout)
        #self.attention = SelfAttention(hidden_dim)           #****************cut this line
        self.flat = nn.Flatten()
        self.fc = nn.Linear(hidden_dim * MAX_LEN, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [sent len, batch size]
        embedded = self.embedding(x)
        # embedded: [sent len, batch size, emb dim]
        output, hidden = self.gru(embedded)
        # use 'batch_first' if you want batch size to be the 1st para
        # output: [sent len, batch size, hid dim*num directions]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output: [sent len, batch size, hid dim]
        output = output.permute(1, 0, 2)
        # ouput: [batch size, sent len, hid dim]
        #new_embed, weights = self.attention(ouput)     #****************cut this line
        # new_embed: [batch size, hid dim]
        # weights: [batch size, sent len]
        output = self.flat(output)
        new_embed = self.dropout(output)   #*************change argument to output
        new_embed = self.fc(new_embed).squeeze()         #****************** squeeze to adjust size
        return new_embed
#-----------------------------------------------------------------------------------------------------------------

#Modified from: https://github.com/PacktPublishing/Mastering-Pytorch/blob/master/Chapter05/transformer.ipynb 

class PosEnc(nn.Module):
    def __init__(self, d_m, dropout=0.2, size_limit=5000):
        # d_m is same as the dimension of the embeddings
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        p_enc = torch.zeros(size_limit, d_m)
        pos = torch.arange(0, size_limit, dtype=torch.float).unsqueeze(1)
        # create list of radians, later multiplied by position indices of words, fed to sine and cosine function
        divider = torch.exp(torch.arange(0, d_m, 2).float() * (-math.log(10000.0) / d_m))
        # pos increases linearly across rows, divider decreases geometrically within each row
        p_enc[:, 0::2] = torch.sin(pos * divider)    #assign alternatively to first, third, fifth etc
        p_enc[:, 1::2] = torch.cos(pos * divider)    #assign alternatively to second, forth, sixth etc
        #note: transpose is after unsqueeze, output size is [5000, 1, d_m]
        p_enc = p_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer('p_enc', p_enc)

    def forward(self, x):
        #final size is [x.size(0), x.size(0), d_m]
        return self.dropout(x + self.p_enc[:x.size(0), :])

    
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, n_layers, MAX_LEN, dropout=0.3):
        super().__init__()
        self.model_name = 'transformer'
        self.mask_source = None
        self.position_enc = PosEnc(embedding_dim, dropout)
        layers_enc = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
        self.enc_transformer = TransformerEncoder(layers_enc, n_layers)
        self.enc = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.dec = nn.Linear(embedding_dim, vocab_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(vocab_size * MAX_LEN, 1)
        self.init_params()

    def _gen_sqr_nxt_mask(self, size):
        msk = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        msk = msk.float().masked_fill(msk == 0, float('-inf'))
        msk = msk.masked_fill(msk == 1, float(0.0))
        return msk

    def init_params(self):
        initial_rng = 0.12
        self.enc.weight.data.uniform_(-initial_rng, initial_rng)
        self.dec.bias.data.zero_()
        self.dec.weight.data.uniform_(-initial_rng, initial_rng)

    def forward(self, source):
        if self.mask_source is None or self.mask_source.size(0) != len(source):
            device = source.device
            msk = self._gen_sqr_nxt_mask(len(source)).to(device)
            self.mask_source = msk

        source = self.enc(source) * math.sqrt(self.embedding_dim)
        source = self.position_enc(source)
        op = self.enc_transformer(source, self.mask_source)
        op = self.dec(op).permute(1,0,2)   #decode and change first dimension to be batch size
        op = self.flatten(op)
        op = self.fc(op).squeeze()
        return op
#------------------------------------------------------------------------------------------------------------------------------