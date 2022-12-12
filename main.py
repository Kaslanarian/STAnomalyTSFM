# %%
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
# %%
from model.anomaly_tsfm import TAnomalyTSFM, SAnomalyTSFM
from util import (
    grid_to_graph,
    period_decay_inject,
    lt_volume_inject,
    lt_attr_inject,
)

np.random.seed(42)
torch.manual_seed(42)
# %%
# f = h5py.File('data/BJ13_M32x32_T30_InOut.h5')

n_region = 10
n_day = 7
start, end = 16 - n_region // 2, 16 + n_region // 2

data = np.load("data/2013.npy")  # (7, 48, 2, 32, 32)
data = data[:n_day, :, :, start:end, start:end]  # (3, 48, 2, 20, 20)
data = data.reshape(*data.shape[:3], -1)  # (3, 48, 2, 400)
data = data.transpose(3, 2, 0, 1)  # (400, 2, 3, 48)
data = data.reshape(*data.shape[:2], -1)  # (400, 2, 3 * 48)
data = data.swapaxes(1, 2)

X, y = lt_attr_inject(data, np.zeros(data.shape[0]), 10)
# X, y = period_decay_inject(data, np.zeros(data.shape[0]), contamination=0.1)
# X, y = lt_volume_inject(
#     data,
#     np.zeros(data.shape[0]),
#     60,
#     contamination=0.1,
# )
data = torch.tensor(X, dtype=torch.float)

edge_index = grid_to_graph((n_region, n_region))
coord = np.array(np.meshgrid(np.arange(n_region),
                             np.arange(n_region))).reshape(2, -1).T

data = data.cuda()
A = to_dense_adj(edge_index)[0].cuda()
D = torch.from_numpy(distance_matrix(coord, coord, p=1)).cuda()


# %%
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class STAnomalyTSFM(nn.Module):

    def __init__(self, D, A, d_in, d_model) -> None:
        super().__init__()

        self.lin_in = nn.Linear(d_in, d_model)
        self.ttsfm = TAnomalyTSFM('long', d_model, 32, 32, 2, dim_fc=128)
        self.stsfm = SAnomalyTSFM(D, A, None, d_model, 32, 32, 2, dim_fc=128)
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


# %%
k1, k2 = 3., 3.
model = STAnomalyTSFM(D, A, 2, 512).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []
for i in range(1000):
    model.train()
    # for i in tqdm(range(100)):
    output, ls1, lp1, ls2, lp2 = model(data)
    # print(ls1.item(), lp1.item(), ls2.item(), lp2.item())
    recon_loss = F.mse_loss(output, data)
    loss1 = recon_loss - k1 * ls1 - k2 * ls2
    loss2 = recon_loss + k1 * lp1 + k2 * lp2

    optimizer.zero_grad()
    loss1.backward(retain_graph=True)
    loss2.backward()
    optimizer.step()

    model.eval()
    score = model.score(data)
    print(i, roc_auc_score(y, score))
    # loss_list.append((loss1.item(), loss2.item()))

# %%
score = model.score(data)
print(roc_auc_score(y, score))
# %%
