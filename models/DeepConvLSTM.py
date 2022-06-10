import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(configs.seq_len, configs.hidden_chn, configs.filter_size, padding='same'),
            nn.BatchNorm1d(configs.hidden_chn)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(configs.hidden_chn, configs.hidden_chn, configs.filter_size, padding='same'),
            nn.BatchNorm1d(configs.hidden_chn)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(configs.hidden_chn, configs.hidden_chn, configs.filter_size, padding='same'),
            nn.BatchNorm1d(configs.hidden_chn)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(configs.hidden_chn, configs.hidden_chn, configs.filter_size, padding='same'),
            nn.BatchNorm1d(configs.hidden_chn)
        )
        self.lstm = nn.LSTM(configs.hidden_chn, configs.n_hidden, configs.n_layers)
        self.fc = nn.Linear(configs.n_hidden, configs.c_out)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = x.float().transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.permute(2, 0, 1)
        h_0 = torch.zeros(2, x.shape[1], 512).cuda()
        c_0 = torch.zeros(2, x.shape[1], 512).cuda()
        x, hidden = self.lstm(x, [h_0, c_0])
        x = x[-1]
        x = self.dropout(x)
        out = self.fc(x)
        return out
    '''
    def parameter(self):
        flops, params = profile(self, inputs=(torch.randn(1, 32, 256)).float().to(device),))
        print('float operations per second:', flops, 'parameters:', params)
    '''

