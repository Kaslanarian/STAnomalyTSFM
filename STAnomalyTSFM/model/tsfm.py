import torch
from torch import nn
import numpy as np
from math import sqrt

from .embed import TemporalEmbedding, SpatialEmbedding


class MultiheadAttention(nn.Module):
    '''
    For temporal input
    '''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k * n_heads)
        self.k = nn.Linear(d_model, dim_k * n_heads)
        self.v = nn.Linear(d_model, dim_v * n_heads)
        self.o = nn.Linear(dim_v * n_heads, d_model)
        self.norm_fact = 1 / sqrt(d_model)

    def generate_mask(self, dim):
        # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
        # padding mask 在数据输入模型之前完成。
        matrix = np.ones((dim, dim))
        mask = torch.Tensor(np.triu(matrix))
        return mask == 1

    def forward(self, x, y, requires_mask=False):
        '''x : (B, L, D)'''
        B, L, _ = x.shape
        Q = self.q(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        K = self.k(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        V = self.v(y).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        scores = torch.einsum("blhe,bshe->bhls", Q, K) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            scores.masked_fill(
                mask,
                value=float("-inf"),
            )
        scores = scores.softmax(dim=-1)
        output = torch.einsum("bhls,bshd->blhd", scores, V).reshape(B, L, -1)
        return self.o(output), scores


class _SingleLayerTemporalTSFM(nn.Module):
    '''无位置编码'''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=dim_fc,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_fc,
                               out_channels=d_model,
                               kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        y = x = self.norm1(self.dropout(x + self.attn(x, x)[0]))
        y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class TemporalTSFM(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embed = TemporalEmbedding(d_in, d_model)
        self.model_list = nn.Sequential(*[
            _SingleLayerTemporalTSFM(d_model, dim_k, dim_v, n_heads, dim_fc)
            for _ in range(n_layers)
        ])
        self.projection = nn.Linear(d_model, d_in)

    def forward(self, x):
        return self.projection(self.model_list(self.embed(x)))


class SpatialTSFM(TemporalTSFM):

    def __init__(
        self,
        edge_index,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        n_layers: int = 1,
    ) -> None:
        super().__init__(d_in, d_model, dim_k, dim_v, n_heads, dim_fc,
                         n_layers)
        self.embed = SpatialEmbedding(edge_index, d_in, d_model)

    def forward(self, x):
        return self.projection(self.model_list(self.embed(x).swapaxes(
            0, 1))).swapaxes(0, 1)
