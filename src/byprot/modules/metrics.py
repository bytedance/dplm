
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
import torch
import torch.nn.functional as F
from functools import partial
import numpy as np


def luost_rmsd(res_list1: list, res_list2: list):
    res_short, res_long = (res_list1, res_list1) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = res_short[i]
        coord_j = res_long[j]
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([d(i + k, j + k) for k in range(N - j)])

    for j in range(N):
        SD[M - 1, j] = d(M - 1, j)

    for i in range(M - 2, -1, -1):
        for j in range((N - (M - i)) - 1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i + 1, j + 1],
                SD[i, j + 1]
            )

    min_SD = SD[0, :N - M + 1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD


def rmsd(pred, target, mask=None):
    assert pred.shape == target.shape
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    rmsd = []
    for p, t, m in zip(pred, target, mask):
        rmsd.append(luost_rmsd(p[m], t[m]))
    return np.mean(rmsd)


def accuracy(pred, target, mask=None, reduction='all'):
    assert pred.shape == target.shape
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    
    return (pred[mask] == target[mask]).sum() / mask.sum()

def accuracy_per_sample(pred, target, mask=None):
    assert pred.shape == target.shape
    bsz = target.shape[0] 

    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    pred = pred.view(bsz, -1)
    target = target.view(bsz, -1)
    mask = mask.view(bsz, -1)

    return ((pred == target) * mask).sum(1) / mask.sum(1)

