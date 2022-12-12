import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from torch_geometric.utils import to_dense_adj
from STAnomalyTSFM.utils.graph import normalize


class TemporalPE(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(TemporalPE, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SpatialPE(nn.Module):

    def __init__(self, edge_index, d_model) -> None:
        super().__init__()
        A = to_dense_adj(edge_index)[0]
        self.A = nn.parameter.Parameter(normalize(A))
        self.linear = nn.Linear(self.A.shape[0], d_model)

    def forward(self, x):
        embed = self.linear(self.A)  # n, d
        return embed.unsqueeze(1)


class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):

    def __init__(self, c_in, d_model, dropout=0.0):
        super(TemporalEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = TemporalPE(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class SpatialEmbedding(nn.Module):

    def __init__(self, edge_index, c_in, d_model, dropout=0.0):
        super(SpatialEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = SpatialPE(
            edge_index=edge_index,
            d_model=d_model,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)