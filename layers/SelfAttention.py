import math

import torch
import torch.nn.functional as F

from torch import nn


class DotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    # queries: (batch_size, query数量, d)
    # keys: (batch_size, key-value pairs数量, d)
    # values: (batch_size, key-value pairs数量, d)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # bmm: batch matrix multiplication
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 分类EEG只用到Encoder，无需使用Decoder，所以不需要mask
        attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, attention_dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.output_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.attention = DotProductAttention(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B * H, L, -1)
        keys = self.key_projection(keys).view(B * H, S, -1)
        values = self.value_projection(values).view(B * H, S, -1)

        output = self.attention(queries, keys, values)
        output = output.view(B, L, -1)

        return self.output_projection(output)


class SelfAttention(nn.Module):
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = nn.Conv1d(n_channels, n_channels//div, 1)
            self.key = nn.Conv1d(n_channels, n_channels//div, 1)
        else:
            self.query = nn.Conv1d(n_channels, n_channels, 1)
            self.key = nn.Conv1d(n_channels, n_channels, 1)
        self.value = nn.Conv1d(n_channels, n_channels, 1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
