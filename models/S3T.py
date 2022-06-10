import torch

from torch import nn

from layers.TransformerEncoder import Encoder, EncoderBlock
from layers.SelfAttention import MultiHeadAttention
from layers.Embedding import PositionalEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.spatial_encoder = Encoder(
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
        self.enc_embedding = PositionalEmbedding(configs.d_model)
        self.temporal_encoder = Encoder(
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
        self.gap = nn.AvgPool1d(configs.seq_len // configs.d_model)
        self.output = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.spatial_encoder(x)
        x = x.transpose(1, 2)
        x = x + self.enc_embedding(x)
        _, _, D = x.shape
        x = torch.mean(x, dim=-1)
        x = x.view(x.shape[0], -1, D)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)
        x = self.gap(x).squeeze(2)
        out = self.output(x)
        return out
