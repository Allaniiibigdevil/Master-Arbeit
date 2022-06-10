import torch.nn.functional as F

from torch import nn


class PreConvolutionalLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, stride):
        super(PreConvolutionalLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=stride, padding='same')
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=stride, padding='same')
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=stride, padding='same')
        self.conv4 = nn.Conv1d(in_channels=hidden_channels, out_channels=input_channels,
                               kernel_size=kernel_size, stride=stride, padding='same')

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.transpose(1, 2)
        return x
