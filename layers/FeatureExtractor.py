from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, filter_num, filter_size, stride, activation):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1), (stride, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), (stride, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), (stride, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), (stride, 1))
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        out = self.activation(self.conv4(x))
        return out
