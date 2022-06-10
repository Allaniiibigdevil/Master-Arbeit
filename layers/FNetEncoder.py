import torch
import torch.nn.functional as F

from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        new_x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.ModuleList(encoder_layer)

    def forward(self, x):
        for encoder_layer in self.encoder_layer:
            x = encoder_layer(x)
        return x
