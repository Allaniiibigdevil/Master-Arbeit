import torch
import torch.nn.functional as F

from torch import nn

from layers.TransformerEncoder import Encoder, EncoderBlock
from layers.SelfAttention import MultiHeadAttention
from layers.Embedding import PositionalEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_embedding = PositionalEmbedding(configs.d_model)
        self.time_encoder = Encoder(
            [
                EncoderBlock(
                    MultiHeadAttention(configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.n_heads,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ]
        )
        # Because of the transpose of input data d_model in channel encoder is the seq_len in original transformer
        self.channel_encoder = Encoder(
            [
                EncoderBlock(
                    MultiHeadAttention(configs.d_spatial_model, configs.n_heads),
                    configs.d_spatial_model,
                    configs.n_heads,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ]
        )
        self.gate = nn.Linear(configs.d_model * configs.seq_len + configs.d_spatial_model * configs.label_len, 2)
        self.output = nn.Linear(configs.d_model * configs.seq_len + configs.d_spatial_model * configs.label_len,
                                configs.c_out)

    def forward(self, x):
        t_input, c_input = x + self.enc_embedding(x), x.transpose(1, 2)
        t_out, c_out = self.time_encoder(t_input), self.channel_encoder(c_input)
        t_out, c_out = t_out.reshape(t_out.shape[0], -1), c_out.reshape(c_out.shape[0], -1)
        gate = F.softmax(self.gate(torch.concat((t_out, c_out), dim=-1)), dim=-1)
        gate = torch.concat((torch.mul(t_out, gate[:, 0:1]), torch.mul(c_out, gate[:, 1:2])), dim=-1)
        out = self.output(gate)
        return out
