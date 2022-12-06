import torch
from torch import nn
import numpy as np
from math import sqrt


class TemporalPE(nn.Module):

    def __init__(self, seq_len: int, embed_dim: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        pe = np.zeros((seq_len, embed_dim))
        pos_col = np.c_[range(seq_len)]
        even = np.sin(pos_col /
                      (10000**(2 * np.arange(0, embed_dim, 2) / embed_dim)))
        odd = np.cos(pos_col /
                     (10000**(2 * np.arange(1, embed_dim, 2) / embed_dim)))
        pe[:, 0::2] = even
        pe[:, 1::2] = odd
        self.pe = torch.from_numpy(pe)

    def forward(self, x):
        if x.ndim == 2:
            return self.pe
        # Batch input
        return self.pe.expand(x.shape[0])


class SpatialPE(nn.Module):

    def __init__(self, A, embed_dim: int) -> None:
        super().__init__()
        self.A = nn.parameter.Parameter(A)
        self.linear = nn.Linear(A.shape[0], embed_dim)

    def forward(self, x):
        embed = self.linear(self.A)
        if x.ndim == 2:
            return embed
        # Batch input
        return embed.expand(
            x.shape[1],
            x.shape[0],
            x.shape[2],
        ).permute(1, 0, 2)


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

        self.q = nn.Linear(d_model, dim_k, bias=False)
        self.k = nn.Linear(d_model, dim_k, bias=False)
        self.v = nn.Linear(d_model, dim_v, bias=False)

        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / sqrt(d_model)

    def generate_mask(self, dim):
        # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
        # padding mask 在数据输入模型之前完成。
        matrix = np.ones((dim, dim))
        mask = torch.Tensor(np.triu(matrix))
        return mask == 1

    def forward(self, x, y, requires_mask=False):
        '''x : (B, L, D)'''
        assert self.dim_k % self.n_heads == 0
        assert self.dim_v % self.n_heads == 0
        Q = self.q(x).reshape(  # (B, L, D, N * K)
            x.shape[0],
            x.shape[1],
            self.n_heads,
            self.dim_k // self.n_heads,
        ).permute(2, 0, 1, 3)  # (N, B, L, K)
        K = self.k(x).reshape(
            x.shape[0],
            x.shape[1],
            self.n_heads,
            self.dim_k // self.n_heads,
        ).permute(2, 0, 1, 3)  # (N, B, L, K)
        V = self.v(y).reshape(
            y.shape[0],
            y.shape[1],
            self.n_heads,
            self.dim_v // self.n_heads,
        ).permute(2, 0, 1, 3)  # (N, B, L, V)
        attention_score = torch.matmul(Q, K.swapaxes(
            2, 3)) * self.norm_fact  # (N, B, L, L)
        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score.masked_fill(
                mask,
                value=float("-inf"),
            )
        attention_score = attention_score.softmax(dim=-1)
        output = torch.matmul(
            attention_score,
            V,
        ).permute(2, 0, 1, 3).reshape(
            y.shape[0],
            y.shape[1],
            -1,
        )  # (N, B, L, V) -> (B, L, N * V)
        output = self.o(output)  # (N, B, D)
        return output, attention_score


class TemporalTSFM(nn.Module):

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_fc),
            nn.ReLU(),
            nn.Linear(dim_fc, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        if not hasattr(self, "pe"):
            self.pe = TemporalPE(*x.shape[1:])
        x = x + self.pe.forward(x)
        x = self.norm1(x + self.attn(x, x)[0])
        return self.norm2(x + self.fc(x))


class SpatialTSFM(TemporalTSFM):

    def __init__(
        self,
        A,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
    ) -> None:
        super().__init__(d_model, dim_k, dim_v, n_heads, dim_fc)
        self.pe = SpatialPE(A, d_model)

    def forward(self, x):
        x = torch.swapaxes(x + self.pe(x), 0, 1)
        x = self.norm1(x + self.attn(x, x)[0])
        return self.norm2(x + self.fc(x)).swapaxes(0, 1)
