import torch

from torch import nn
from einops import repeat

from layers.SelfAttention import SelfAttention, MultiHeadAttention
from layers.TransformerEncoder import Encoder, EncoderBlock


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.window_size = configs.window_size
        self.window_stride = configs.window_stride
        self.window_num = (configs.seq_len - configs.window_size) // configs.window_stride + 1
        self.pool = configs.pool
        # self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model * configs.window_size * 2))
        self.cls_token = nn.Parameter(torch.randn(1, 1, configs.window_size * 2))

        self.attn = SelfAttention(configs.window_size * 2, div=1)
        self.dropout = nn.Dropout(configs.dropout)
        '''
        self.encoder = Encoder(
            [
                EncoderBlock(
                    MultiHeadAttention(configs.d_model * configs.window_size * 2, configs.n_heads),
                    configs.d_model * configs.window_size * 2,
                    configs.n_heads,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.d_model * configs.window_size * 2, configs.c_out, bias=True)
        '''
        self.encoder = Encoder(
            [
                EncoderBlock(
                    MultiHeadAttention(configs.window_size * 2, configs.n_heads),
                    configs.window_size * 2,
                    configs.n_heads,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.window_size * 2, configs.c_out, bias=True)

    def forward(self, x):
        x = torch.stack([
            x[:, self.window_stride * t: self.window_stride * t + self.window_size, :] for t in range(self.window_num)
        ], dim=1)
        # x: (batch_size, new_length, window_size, feature) after deformation
        x = torch.cat([torch.fft.fft(x, dim=2).real, torch.fft.fft(x, dim=2).imag], dim=2)
        refined = torch.cat(
            [self.attn(torch.unsqueeze(x[:, t, :, :], dim=3)) for t in range(x.shape[1])],
            dim=-1
        )
        # refine: (batch_size, window_size * 2, feature, new_length)
        x = refined.permute(0, 3, 1, 2)
        # x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.mean(dim=3)
        B, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        enc_out = self.encoder(x)
        enc_out = enc_out.mean(dim=1) if self.pool == 'mean' else enc_out[:, 0]
        out = self.projection(enc_out.view(B, -1))
        return out
