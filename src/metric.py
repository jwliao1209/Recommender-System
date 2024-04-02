import numpy as np


def cal_mae(X, Y, N):
    return np.sum(np.abs(X - Y)) / N


def cal_rmse(X, Y, N):
    return np.sqrt(np.sum((X - Y) ** 2) / N)
