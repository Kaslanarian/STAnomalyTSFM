import torch
from torch import nn
from math import sqrt, pi
from typing import Literal

from tsfm import TemporalTSFM, SpatialTSFM, TemporalPE


class ShortAnomalyAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.sigma_projection = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x):
        '''x     : (B, L, d_model)'''
        '''sigma : (B, L, n_heads)'''
        sigma = self.sigma_projection(x).transpose(1, 2)  # (B, H, L)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, sigma.shape[-1])  # B H L L
        prior = 1.0 / (sqrt(2 * pi) * sigma) * torch.exp(
            -self.distances(sigma.shape[-1]).to(sigma.device)**2 /
            (2 * (sigma**2)))
        return prior / prior.sum(-1, keepdims=True)  # (B, H, L, L)

    def distances(self, L):
        arange = torch.arange(L)
        return (arange.reshape(-1, 1) - arange).abs()


class LongAnomalyAttention(ShortAnomalyAttention):

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__(d_model, n_heads)
        self.p_projection = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x):
        '''
        x     : (B, L, d_model)
        sigma : (B, L, n_heads)
        p     : (B, L, n_heads)
        '''
        sigma = self.sigma_projection(x).transpose(1, 2)  # (B, H, L)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, sigma.shape[-1])  # B H L L

        p = self.p_projection(x).transpose(1, 2)  # (B, H, L)
        p = torch.sigmoid(p * 5) + 1e-5
        p = torch.pow(3, p) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, p.shape[-1])  # B H L L

        x = self.distances(sigma.shape[-1]).to(sigma.device)
        sigma2 = sigma**2
        prior = torch.exp(-x**2 /
                          (2 * sigma2) - torch.cos(pi * x / p**2) / sigma2) / (
                              sqrt(2 * pi) * sigma)
        return prior / prior.sum(-1, keepdims=True)  # (B, H, L, L)


class SpatialAnomalyAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        D,
        A,
        POI=None,
    ) -> None:
        super().__init__()
        self.sigma1_projection = nn.Linear(d_model, n_heads, bias=False)
        self.sigma2_projection = nn.Linear(d_model, n_heads, bias=False)

        self.D = D
        self.A = A
        self.POI = POI

    def forward(self, x):
        '''
        x : (B, L, H)
        '''
        sigma1 = self.sigma1_projection(x).swapaxes(0, 2)  # (L, H, B)
        sigma1 = torch.sigmoid(sigma1 * 5) + 1e-5
        sigma1 = torch.pow(3, sigma1) - 1
        sigma1 = sigma1.unsqueeze(-1).repeat(1, 1, 1,
                                             sigma1.shape[-1])  # L H B B
        sigma2 = self.sigma2_projection(x).transpose(1, 2)  # (L, H, B)
        sigma2 = torch.sigmoid(sigma2 * 5) + 1e-5
        sigma2 = torch.pow(3, sigma2) - 1
        sigma2 = sigma2.unsqueeze(-1).repeat(1, 1, 1,
                                             sigma2.shape[-1])  # L, H, B, B

        sum_list = []
        prior1 = 1.0 / (sqrt(2 * pi) * sigma1) * torch.exp(
            -self.D.to(sigma1.device)**2 / (2 * (sigma1**2)))
        sum_list.append(prior1 / prior1.sum(-1, keepdims=True))

        if self.POI is not None:
            prior2 = 1.0 / (sqrt(2 * pi) * sigma2) * torch.exp(
                -self.POI.to(sigma2.device)**2 / (2 * (sigma2**2)))
            sum_list.append(prior2 / prior2.sum(-1, keepdims=True))

        prior3 = self.A.unsqueeze(0).unsqueeze(0).repeat(
            sigma1.shape[0],
            sigma1.shape[1],
            1,
            1,
        ).to(sigma1.device)
        sum_list.append(prior3 / prior3.sum(-1, keepdims=True))

        return sum(sum_list) / len(sum_list)  # L, H, B, B


class TAnomalyTSFM(TemporalTSFM):

    def __init__(
        self,
        type: Literal["short", 'long'],
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
    ) -> None:
        super().__init__(d_model, dim_k, dim_v, n_heads, dim_fc)
        self.anomaly_attn = {
            "short": ShortAnomalyAttention,
            "long": LongAnomalyAttention,
        }[type](d_model, n_heads)

    def forward(self, x):
        if not hasattr(self, "pe"):
            self.pe = TemporalPE(*x.shape[1:])
        prior = self.anomaly_attn(x)
        x = x + self.pe.forward(x)
        output, series = self.attn(x, x)
        x = self.norm1(x + output)
        output = self.norm2(x + self.fc(x))
        return output, prior, series


class SAnomalyTSFM(SpatialTSFM):

    def __init__(
        self,
        D,
        A,
        POI,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
    ) -> None:
        super().__init__(A, d_model, dim_k, dim_v, n_heads, dim_fc)
        self.anomaly_attn = SpatialAnomalyAttention(
            d_model,
            n_heads,
            D,
            A,
            POI,
        )

    def forward(self, x):
        prior = self.anomaly_attn(x)  # (L, H, B, B)
        x = torch.swapaxes(x + self.pe(x), 0, 1)  # (L, B, H)
        output, series = self.attn(x, x)  # series (N, L, B, B)
        x = self.norm1(x + output)
        output = self.norm2(x + self.fc(x)).swapaxes(0, 1)
        return output, prior, series
