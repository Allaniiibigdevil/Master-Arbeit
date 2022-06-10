from torch import nn

from layers.FNetEncoder import Encoder, EncoderBlock
from layers.Embedding import PositionalEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_embedding = PositionalEmbedding(configs.d_model)
        self.encoder = Encoder(
            [
                EncoderBlock(
                    configs.d_model,
                    configs.d_ff,
                    configs.dropout,
                    configs.activation
                ) for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.seq_len * configs.d_model, configs.c_out, bias=True)

    def forward(self, x):
        enc_out = x + self.enc_embedding(x)
        enc_out = self.encoder(enc_out)
        B, _, _ = enc_out.shape
        enc_out = self.projection(enc_out.view(B, -1))
        return enc_out
