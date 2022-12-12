# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.lof import LocalOutlierFactor
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from STAnomalyTSFM.utils.inject import *

n_region = 20
n_day = 5
start, end = 16 - n_region // 2, 16 + n_region // 2

data = np.load("data/2013.npy")  # (7, 48, 2, 32, 32)
data = data[:n_day, :, :, start:end, start:end]  # (3, 48, 2, 20, 20)
data = data.reshape(*data.shape[:3], -1)  # (3, 48, 2, 400)
data = data.transpose(3, 2, 0, 1)  # (400, 2, 3, 48)
data = data.reshape(*data.shape[:2], -1)  # (400, 2, 3 * 48)
data = data.swapaxes(1, 2)
data = (data - data.min(0)) / (data.max(0) - data.min(0))

l = []
for i in tqdm(range(20)):
    # X, y = lt_volume_inject(
    #     data,
    #     np.zeros(data.shape[0]),
    #     contamination=0.1,
    # )
    # X, y = lt_volume_inject(data, np.zeros(data.shape[0]), threshold=60)

    # X, y = lt_attr_inject(data, np.zeros(data.shape[0]), n_region)
    X, y = lt_volume_inject(data, np.zeros(data.shape[0]), 0.5)
    # X, y = period_decay_inject(data,
    #                            np.zeros(data.shape[0]),
    #                            contamination=0.1)
    # X, y = lt_inout_inject(data, np.zeros(data.shape[0]), contamination=0.1)
    # X, y = lt_volume_inject(data,
    #                         np.zeros(data.shape[0]),
    #                         contamination=0.1,
    #                         ratio=2.)

    X = X.reshape(X.shape[0], -1)
    lof = LocalOutlierFactor(novelty=True).fit(X)
    ocsvm = OCSVM().fit(X)
    iforest = IForest(n_jobs=5).fit(X)

    l.append([
        roc_auc_score(y, -lof.decision_function(X)),
        roc_auc_score(y, ocsvm.decision_function(X)),
        roc_auc_score(y, iforest.decision_function(X)),
    ])

np.array(l).mean(0)
# %%
data.shape
# %%
plt.plot(data[18, :48])
# %%
data.shape
# %%
plt.hist(data.sum((1, 2)))
# %%
data = np.load("data/2013.npy")
data.shape
# %%
for i in range(7):
    d = data[i, :, 0, 2, 2]
    threshold = np.percentile(d, 60)
    plt.plot(np.minimum(d, threshold), label='day {}'.format(i + 1))

plt.legend()
# %%
