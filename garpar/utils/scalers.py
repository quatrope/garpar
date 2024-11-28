# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Scalers."""

import numpy as np


def proportion_scaler(arr):
    """
    Scale the input array by dividing each element by the sum of the array.

    Parameters
    ----------
    arr : array-like
        Input array to be scaled.

    Returns
    -------
    array-like
        Scaled array where each element is divided by the sum of the parameter.
    """
    arr = np.asarray(arr)
    asum = np.sum(arr)
    return arr / asum


def minmax_scaler(arr):
    """
    Scales the input array using the min-max normalization technique.

    Parameters
    ----------
    arr : array-like
        The input array to be scaled.

    Returns
    -------
    array-like
        The scaled array where each element is normalized to the range [0, 1]
        based on the minimum and maximum values of the input array.
    """
    arr = np.asarray(arr)
    amin, amax = np.min(arr), np.max(arr)
    return (arr - amin) / (amax - amin)


def max_scaler(arr):
    """
    Scales the input array to have the maximum value in the array as the
    scaler.

    Parameters
    ----------
    arr : array-like
        The input array to be scaled.

    Returns
    -------
    array-like
        The scaled array where each element is divided by the maximum value of
        the input array.

    Notes
    -----
    This function uses the `numpy.max` function to find the maximum value in
    the input array. The input array is then divided element-wise by the
    maximum value, resulting in a scaled array.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> max_scaler(arr)
    array([0.2 , 0.4 , 0.6 , 0.8 , 1.  ])
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
