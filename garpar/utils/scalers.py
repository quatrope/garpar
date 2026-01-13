# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2026
#   Lautaro Ebner,
#   Diego Gimenez,
#   Nadia Luczywo,
#   Juan Cabral,
#   and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Scalers."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


# =============================================================================
# SCALERS
# =============================================================================


def proportion_scaler(arr):
    """Scale the input array by normalizing each element."""
    arr = np.asarray(arr)
    asum = np.sum(arr)
    return arr / asum


def minmax_scaler(arr):
    """Scales the input array using the min-max normalization technique."""
    arr = np.asarray(arr)
    amin, amax = np.min(arr), np.max(arr)
    return (arr - amin) / (amax - amin)


def max_scaler(arr):
    """Scales the input array by the maximun value."""
    arr = np.asarray(arr)
    amax = np.max(arr)
    return arr / amax


def standar_scaler(arr):
    """Standardize an array by subtracting the mean and dividing by std."""
    arr = np.asarray(arr)
    mu, sigma = np.mean(arr), np.std(arr)
    return (arr - mu) / sigma
