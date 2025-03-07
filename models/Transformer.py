import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ]
        )

        # second encoder
        self.fc = nn.Linear(configs.d_model*self.charge_discharge_length, configs.d_model)
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ]
        )
        
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * self.early_cycle_threshold, configs.output_num)

    def classification(self, x_enc, curve_attn_mask):
        # Embedding
        B, L = x_enc.shape[0], x_enc.shape[1]
        x_enc = x_enc.reshape(B*L, -1, self.charge_discharge_length)
        x_enc = x_enc.transpose(1, 2)
        
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        enc_out = enc_out.reshape(B*L, -1)
        enc_out = self.fc(enc_out) # [B*L, d_model]
        enc_out = enc_out.reshape(B, L, -1)
        
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L]
        curve_attn_mask = torch.repeat_interleave(curve_attn_mask, curve_attn_mask.shape[-1], dim=1) # [B, L, L]
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L, L]
        curve_attn_mask = curve_attn_mask==0 # set True to mask
        output, attns = self.encoder2(enc_out, attn_mask=curve_attn_mask)

        # Output
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self,  cycle_curve_data, curve_attn_mask):
        '''
        params:
            cycle_curve_data: [B, L, num_variables, fixed_length_of_curve]
            curve_attn_mask: [B, L]
        '''
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
        dec_out = self.classification(cycle_curve_data, curve_attn_mask)
        return dec_out  # [B, N]
