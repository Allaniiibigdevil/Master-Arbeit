from torch import nn

from layers.TransformerEncoder import Encoder, EncoderBlock
from layers.SelfAttention import MultiHeadAttention
from layers.Embedding import PositionalEmbedding
from layers.PreCNN import PreConvolutionalLayer


class Model(nn.Module):
    """
    Transformer Encoder for EEG Classification
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.cnn = PreConvolutionalLayer(configs.label_len, configs.hidden_chn, configs.kernel_size, configs.stride)
        self.enc_embedding = PositionalEmbedding(configs.d_model)
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
        self.projection = nn.Linear(configs.seq_len * configs.d_model, configs.c_out, bias=True)

    def forward(self, x):
        x = self.cnn(x)
        enc_out = x + self.enc_embedding(x)
        enc_out = self.encoder(enc_out)
        B, _, _ = enc_out.shape
        enc_out = self.projection(enc_out.view(B, -1))
        return enc_out
