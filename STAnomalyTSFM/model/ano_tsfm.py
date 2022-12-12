import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt, pi
from typing import Literal
from scipy.special import softmax

from .tsfm import TemporalTSFM, SpatialTSFM, _SingleLayerTemporalTSFM
from .embed import TemporalEmbedding, SpatialEmbedding
from ..utils.graph import normalize


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class ShortAnomalyAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.sigma_projection = nn.Linear(d_model, n_heads)

    def forward(self, x):
        '''x     : (B, L, d_model)'''
        '''sigma : (B, L, n_heads)'''
        sigma = self.sigma_projection(x).transpose(1, 2)  # (B, H, L)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, sigma.shape[-1])  # B H L L
        prior = 1.0 / (sqrt(2 * pi) * sigma) * torch.exp(
            -self.distances(sigma.shape[-1]).to(sigma.device)**2 /
            (2 * torch.square(sigma)))
        return prior / prior.sum(-1, keepdims=True)  # (B, H, L, L)

    def distances(self, L):
        arange = torch.arange(L)
        return (arange.reshape(-1, 1) - arange).abs()


class LongAnomalyAttention(ShortAnomalyAttention):

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__(d_model, n_heads)
        self.p_projection = nn.Linear(d_model, n_heads)

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
        p = 24 * torch.sigmoid(p * 5)
        p = p.unsqueeze(-1).repeat(1, 1, 1, p.shape[-1])  # B H L L

        x = self.distances(sigma.shape[-1]).to(sigma.device)
        prior = 1.0 / (sqrt(2 * pi) * sigma) * torch.exp(
            (-x**2 / 2 + torch.cos(pi * x / p) - 1) / torch.square(sigma))
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
        self.sigma1_projection = nn.Linear(d_model, n_heads)
        self.sigma2_projection = nn.Linear(d_model, n_heads)

        self.D = D
        self.A = A
        self.norm_A = normalize(A).unsqueeze(0).unsqueeze(0)
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
        prior1 = 1.0 / (sqrt(2 * pi) * sigma1) * torch.exp(-self.D**2 /
                                                           (2 * (sigma1**2)))
        return prior1 / prior1.sum(-1, keepdims=True)
        sum_list.append(prior1 / prior1.sum(-1, keepdims=True))

        if self.POI is not None:
            prior2 = 1.0 / (sqrt(2 * pi) * sigma2) * torch.exp(-self.POI**2 /
                                                               (2 *
                                                                (sigma2**2)))
            sum_list.append(prior2 / prior2.sum(-1, keepdims=True))

        prior3 = self.norm_A.repeat(
            sigma1.shape[0],
            sigma1.shape[1],
            1,
            1,
        )
        sum_list.append(prior3 / (prior3.sum(-1, keepdims=True) + 1e-8))

        return sum(sum_list) / len(sum_list)  # L, H, B, B


class _SingleLayerTAnomalyTSFM(_SingleLayerTemporalTSFM):
    '''无位置编码'''

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
        prior = self.anomaly_attn(x)
        output, series = self.attn(x, x)
        y = x = self.norm1(self.dropout(x + output))
        y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), series, prior

    def ass_dis(self, series, prior):
        series_loss = torch.mean(
            my_kl_loss(series, prior.detach()) +
            my_kl_loss(prior.detach(), series))
        prior_loss = torch.mean(
            my_kl_loss(prior, series.detach()) +
            my_kl_loss(series.detach(), prior))
        return series_loss, prior_loss


class TAnomalyTSFM(nn.Module):

    def __init__(
        self,
        type: Literal["short", 'long'],
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
        self.model_list = nn.ModuleList([
            _SingleLayerTAnomalyTSFM(type, d_model, dim_k, dim_v, n_heads,
                                     dim_fc) for _ in range(n_layers)
        ])
        self.projection = nn.Linear(d_model, d_in)

    def forward(self, x):
        series_loss_list = []
        prior_loss_list = []
        x = self.embed(x)
        for model in self.model_list:
            x, series, prior = model(x)
            series_loss, prior_loss = model.ass_dis(series, prior)
            series_loss_list.append(torch.mean(series_loss))
            prior_loss_list.append(torch.mean(prior_loss))

        return (
            self.projection(x),
            sum(series_loss_list) / len(series_loss_list),
            sum(prior_loss_list) / len(prior_loss_list),
        )

    @torch.no_grad()
    def score(self, x):
        criterion = nn.MSELoss(reduction='none')
        series_loss = 0.
        prior_loss = 0.
        x_ = x
        x = self.embed(x)
        for model in self.model_list:
            x, series, prior = model(x)
            series_loss += my_kl_loss(series, prior)
            prior_loss += my_kl_loss(prior, series)
        x = self.projection(x)
        loss = torch.mean(criterion(x_, x), dim=-1)
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        return cri.detach().cpu().numpy().sum(1)


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
        x = x.swapaxes(0, 1)
        output, series = self.attn(x, x)  # series (N, L, B, B)
        x = self.norm1(x + output)
        output = self.norm2(x + self.fc(x)).swapaxes(0, 1)
        return output, prior.swapaxes(0, 1), series


class STAnomalyTSFM(nn.Module):

    def __init__(self, type: str, D, A, P, d_in, d_model, dim_k, dim_v,
                 n_heads, dim_fc) -> None:
        super().__init__()

        self.lin_in = nn.Linear(d_in, d_model)
        self.ttsfm = TAnomalyTSFM(type, d_model, dim_k, dim_v, n_heads, dim_fc)
        self.stsfm = SAnomalyTSFM(D, A, P, d_model, dim_k, dim_v, n_heads,
                                  dim_fc)
        self.lin_out = nn.Linear(d_model, d_in)

    def forward(self, x):
        x = self.lin_in(x)
        output_t, prior_t, series_t = self.ttsfm(x)
        output_s, prior_s, series_s = self.stsfm(output_t + x)
        return self.lin_out(output_s + output_t), *self.ass_dis(
            series_t, prior_t), *self.ass_dis(series_s, prior_s)

    def ass_dis(self, series, prior):
        series_loss = torch.mean(
            my_kl_loss(series, prior.detach()) +
            my_kl_loss(prior.detach(), series))
        prior_loss = torch.mean(
            my_kl_loss(prior, series.detach()) +
            my_kl_loss(series.detach(), prior))
        return series_loss, prior_loss

    @torch.no_grad()
    def score(self, x):
        x_ = self.lin_in(x)
        output_t, prior_t, series_t = self.ttsfm(x_)
        output_s, prior_s, series_s = self.stsfm(output_t + x_)
        tem_dis = my_kl_loss(series_t, prior_t) + my_kl_loss(prior_t, series_t)
        spa_dis = my_kl_loss(series_s, prior_s) + my_kl_loss(prior_s, series_s)
        rec_loss = F.mse_loss(self.lin_out(output_s + output_t),
                              x,
                              reduction='none')

        tem_dis = tem_dis.cpu().numpy()
        spa_dis = spa_dis.cpu().numpy()
        rec_loss = rec_loss.cpu().numpy().mean(-1)
        metric1 = softmax(-spa_dis, axis=-1)  # spatial
        metric2 = softmax(-tem_dis, axis=-1)  # temporal
        return (metric1 * metric2.T * rec_loss.T).sum(0)