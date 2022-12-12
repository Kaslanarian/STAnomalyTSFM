from pyod.models.base import BaseDetector
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score
from typing import Literal

from ..model.tsfm import SpatialTSFM, TemporalTSFM
from ..model.ano_tsfm import SAnomalyTSFM, TAnomalyTSFM
from .util import predict_by_score

import seaborn as sns
import matplotlib.pyplot as plt


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, x):
        super(SimpleDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class TemporalTSFMModel(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc_expand: int,
        n_layers: int = 1,
        device: str = 'cpu',
        epoch: int = 10,
        lr: float = 1e-4,
        batch_size: int = -1,
        term: Literal["long", "short"] = 'long',
        contamination=0.1,
        verbose=False,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.dim_fc_expand = dim_fc_expand
        self.n_layers = n_layers
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.term = term
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "n_heads": n_heads,
            "dim_fc": n_heads * self.d_model,
            "n_layers": n_layers,
        }

    def fit(self, x, edge_index, y=None):
        x_ = torch.tensor(x, dtype=torch.float)
        self.model = TemporalTSFM(**self.tsfm_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = self.batch_size if self.batch_size != -1 else x.shape[0]
        dataset = SimpleDataset(x_)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.epoch):
            for data in dataloader:
                data = data.to(self.device)
                output = self.model(data)
                score = torch.square(output - data).mean((1, 2))
                loss = score.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                with torch.no_grad():
                    cuda_x = x_.to(self.device)
                    output = self.model(cuda_x)
                    recon = torch.square(output - cuda_x)
                    if self.term == 'long':
                        score = recon.mean((1, 2)).cpu().numpy()
                    else:
                        score = recon.mean(-1).flatten().cpu().numpy()
                loss = score.mean()
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss)
                if y is not None:
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output = self.model(x)
        recon = torch.square(output - x)
        if self.term == 'long':
            score = recon.mean((1, 2))
        else:
            score = recon.mean(-1).flatten()
        return score.cpu().numpy()


class TAnomalyTSFMModel(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc_expand: int,
        type: str = 'short',
        n_layers: int = 1,
        device: str = 'cpu',
        k: int = 3,
        epoch: int = 10,
        lr: float = 1e-4,
        batch_size: int = -1,
        contamination=0.1,
        verbose=False,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.dim_fc_expand = dim_fc_expand
        self.type = type
        self.n_layers = n_layers
        self.device = device
        self.k = k
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.tsfm_args = {
            "type": type,
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "n_heads": n_heads,
            "dim_fc": n_heads * self.d_model,
            "n_layers": n_layers,
        }

    def fit(self, x, edge_index, y=None):
        x_ = torch.tensor(x, dtype=torch.float)
        self.model = TAnomalyTSFM(**self.tsfm_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = self.batch_size if self.batch_size != -1 else x.shape[0]
        dataset = SimpleDataset(x_)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.epoch):
            for data in dataloader:
                data = data.to(self.device)
                output, series_loss, prior_loss = self.model(data)
                recon_loss = F.mse_loss(output, data)
                loss1 = recon_loss - self.k * series_loss
                loss2 = recon_loss + self.k * prior_loss

                optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                loss2.backward()
                optimizer.step()

            if self.verbose:
                with torch.no_grad():
                    cuda_x = x_.to(self.device)
                    output, series_loss, prior_loss = self.model(cuda_x)
                    score = self.model.score(cuda_x)
                    loss = F.mse_loss(output, cuda_x) - self.k * series_loss
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        return self.model.score(x)


class SpatialTSFMModel(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc_expand: int,
        n_layers: int = 1,
        device: str = 'cpu',
        epoch: int = 10,
        lr: float = 1e-4,
        batch_size: int = -1,
        contamination=0.1,
        verbose=False,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.dim_fc_expand = dim_fc_expand
        self.n_layers = n_layers
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "n_heads": n_heads,
            "dim_fc": n_heads * self.d_model,
            "n_layers": n_layers,
        }

    def fit(self, x, edge_index, y=None):
        x_ = torch.tensor(x, dtype=torch.float)
        self.model = SpatialTSFM(edge_index, **self.tsfm_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = self.batch_size if self.batch_size != -1 else x.shape[1]
        dataset = SimpleDataset(x_.swapaxes(0, 1))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.model.train()

        for epoch in range(self.epoch):
            for data in dataloader:
                data = data.to(self.device).swapaxes(0, 1)
                output = self.model(data)
                score = torch.square(output - data).mean((1, 2))
                loss = score.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                with torch.no_grad():
                    cuda_x = x_.to(self.device)
                    output = self.model(cuda_x)
                    score = torch.square(output - cuda_x).mean(
                        (1, 2)).cpu().numpy()
                loss = score.mean()
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss)
                if y is not None:
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output = self.model(x)
        score = torch.square(output - x).mean((1, 2))
        return score.cpu().numpy()