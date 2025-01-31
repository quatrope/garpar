# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Scalers."""

import numpy as np


def proportion_scaler(arr):
    """
    Scale the input array by dividing each element by the sum of the array.
    """
    arr = np.asarray(arr)
    asum = np.sum(arr)
    return arr / asum


def minmax_scaler(arr):
    """
    Scales the input array using the min-max normalization technique.
    """
    arr = np.asarray(arr)
    amin, amax = np.min(arr), np.max(arr)
    return (arr - amin) / (amax - amin)


def max_scaler(arr):
    """
    Scales the input array to have the maximum value in the array as the
    scaler.
    """
    arr = np.asarray(arr)
    amax = np.max(arr)
    return arr / amax


def standar_scaler(arr):
    """
    Standardize an array by subtracting the mean and dividing by the standard
    deviation.
    """
    arr = np.asarray(arr)
    mu, sigma = np.mean(arr), np.std(arr)
    return (arr - mu) / sigma
