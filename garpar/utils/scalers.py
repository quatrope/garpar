# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import numpy as np


def proportion_scaler(arr):
    arr = np.asarray(arr)
    asum = np.sum(arr)
    return arr / asum


def minmax_scaler(arr):
    arr = np.asarray(arr)
    amin, amax = np.min(arr), np.max(arr)
    return (arr - amin) / (amax - amin)


def max_scaler(arr):
    arr = np.asarray(arr)
    amax = np.max(arr)
    return arr / amax


def standar_scaler(arr):
    arr = np.asarray(arr)
    mu, sigma = np.mean(arr), np.std(arr)
    return (arr - mu) / sigma
