import numpy as np
from STAnomalyTSFM.utils.inject import *
from STAnomalyTSFM.utils.graph import grid_to_graph
from STAnomalyTSFM.interface.estimator import TemporalTSFMModel, TAnomalyTSFMModel, SpatialTSFMModel

np.random.seed(42)

n_region = 20
n_day = 5
start, end = 16 - n_region // 2, 16 + n_region // 2
edge_index = grid_to_graph((n_region, n_region))

data = np.load("data/2014.npy")  # (7, 48, 2, 32, 32)
data = data[:n_day, :, :, start:end, start:end]  # (3, 48, 2, 20, 20)
data = data.reshape(*data.shape[:3], -1)  # (3, 48, 2, 400)
data = data.transpose(3, 2, 0, 1)  # (400, 2, 3, 48)
data = data.reshape(*data.shape[:2], -1)  # (400, 2, 3 * 48)
data = data.swapaxes(1, 2)

data = (data - data.min(0)) / (data.max(0) - data.min(0))
X, y = period_decay_inject(data, np.zeros(data.shape[0]), contamination=0.1)
# X, y = lt_volume_inject(data, np.zeros(data.shape[0]), 0.5)
# X, y = lt_attr_inject(data, np.zeros(data.shape[0]), k=n_region)

# import matplotlib.pyplot as plt
# x = X[y == 1][:48, :, 0]
# plt.plot(x)
# plt.show()

# X, y = lt_volume_inject(data, np.zeros(data.shape[0]))
model = TAnomalyTSFMModel(
    type='long',
    d_in=2,
    d_model=16,
    dim_k=32,
    dim_v=32,
    n_heads=8,
    dim_fc_expand=2,
    n_layers=1,
    device='cuda',
    epoch=50,
    batch_size=16,
    lr=0.01,
    verbose=True,
).fit(X, edge_index, y)

# %%
# from sagod import models
# from torch_geometric.data import Data
# import torch
# model = models.AnomalyDAE(epoch=100, verbose=True).fit(
#     Data(torch.tensor(X.reshape(X.shape[0], -1), dtype=torch.float),
#          edge_index=edge_index),
#     y,
# )
