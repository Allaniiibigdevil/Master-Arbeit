import torch

from torch import nn
from einops import repeat

from layers.FeatureExtractor import FeatureExtractor
from layers.SelfAttention import SelfAttention, MultiHeadAttention
from layers.TransformerEncoder import Encoder, EncoderBlock


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pool = configs.pool
        # self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model * configs.filter_num))
        self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model))

        self.fe = FeatureExtractor(configs.filter_num, configs.filter_size, configs.fe_stride, configs.activation)
        self.attn = SelfAttention(configs.filter_num, div=1)
        self.dropout = nn.Dropout(configs.dropout)
        '''
        self.encoder = Encoder(
            [
                EncoderBlock(
                    MultiHeadAttention(configs.d_model * configs.filter_num, configs.n_heads),
                    configs.d_model * configs.filter_num,
                    configs.n_heads,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.d_model * configs.filter_num, configs.c_out, bias=True)
        '''
        self.encoder = Encoder(
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
        self.projection1 = nn.Linear(configs.filter_num * configs.d_model, configs.d_model, bias=True)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x):
        # input dimension: (batch_size, length, feature)
        x = self.fe(x)
        # x: (batch_size, channel, new_length, feature) after feature extraction
        refined = torch.cat(
            [self.attn(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1
        )
        # refine: (batch_size, channel, feature, new_length)
        x = refined.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.projection1(x)
        # x = x.mean(dim=3)
        B, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        enc_out = self.encoder(x)
        enc_out = enc_out.mean(dim=1) if self.pool == 'mean' else enc_out[:, 0]
        out = self.projection(enc_out.view(B, -1))
        return out


'''
1. 最后的输出层需要利用class token或者求平均进行改进
2. 改进attn层，加上残差连接
3. 先写introduction(Background, Problem Description, Idea 引出下一章需要读的文章), related work, setup
4. 模型三步：local information - cross channel attention - temporal attention

周五五点半
'''