from torch import nn

from layers.Embedding import PositionalEmbedding
from layers.AutoCorrelation import AutoCorrection, AutoCorrelationLayer
from layers.AutoformerEncoder import Encoder, EncoderLayer, SeriesDecomposition


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.decomp = SeriesDecomposition(configs.moving_avg)
        self.enc_embedding = PositionalEmbedding(configs.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(AutoCorrection(configs.factor, configs.dropout),
                                         configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
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
