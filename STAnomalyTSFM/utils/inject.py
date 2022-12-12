import numpy as np
import torch.nn.functional as F


def period_decay_inject(
    X,
    y,
    contamination: float = 0.1,
):
    '''X : (N, T, D)'''
    X, y = np.copy(X), np.copy(y)
    for _ in range(int(contamination * X.shape[0])):
        while True:
            i = np.random.randint(0, X.shape[0])
            if y[i] == 0:
                y[i] = 1
                X[i] = np.roll(X[i], 2, axis=0)
                break
    return X, y


def lt_volume_inject(
    X,
    y,
    threshold: int = 0.5,
    contamination: float = 0.1,
):
    X, y = np.copy(X), np.copy(y)
    M = X.max()
    for _ in range(int(contamination * X.shape[0])):
        while True:
            i = np.random.randint(0, X.shape[0])
            if y[i] == 0 and X[i].max() > M * threshold:
                y[i] = 1
                t = X[i].max() * threshold
                X[i] = np.minimum(X[i], t)
                break
    return X, y


def lt_attr_inject(
    X,
    y,
    k: int = 50,
    contamination: float = 0.1,
):
    '''
    Attibuted anomaly injection. 
    We randomly choose a point x_i waited to be injected and set C with k points for p times,
    and conduct 
    '''
    X, y = np.copy(X), np.copy(y)
    N, n = X.shape[0], int(contamination * X.shape[0])
    for _ in range(n):
        while True:
            i = np.random.randint(0, N)
            if y[i] == 0:
                rnd_choice = np.random.choice(N, k, replace=False)
                j = np.argmax(np.square(X[rnd_choice] - X[i]).sum((1, 2)))
                X[i] = X[rnd_choice[j]]
                y[i] = 1
                break
    return X, y