# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Entropy."""

from collections import Counter

import numpy as np

import pypfopt

from scipy import stats

import warnings


def _computeMarks(prices, **kwargs):
    """
    Compute the marks for a given set of prices based on returns.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame containing price data with assets as columns.
    **kwargs
        Additional keyword arguments to pass to returns_from_prices.

    Returns
    -------
    np.ndarray
        A binary array where 0 represents returns below the average
        and 1 represents returns above or equal to the average.
    """
    returns = pypfopt.expected_returns.returns_from_prices(
        prices=prices, **kwargs
    )

    avg_returns = returns.mean(axis=0)

    marks = (returns.values >= avg_returns.values).astype(int)

    return marks


def shannon(prices, window_size=None, **kwargs):
    """Calculate the Shannon entropy of the given prices.

    Parameters
    ----------
    prices : array_like
        Prices data to calculate entropy.
    window_size : int, optional
        Ignored parameter for Shannon entropy calculation.
    **kwargs
        Additional keyword arguments to pass to stats.entropy.

    Returns
    -------
    array_like
        The Shannon entropy of the prices along axis 0.
    """
    if window_size is not None:
        warnings.warn(
            f"'window_size={window_size}' is ignored in shannon entropy"
        )
    return stats.entropy(prices, axis=0, **kwargs)


def risso(prices, window_size=None, **kwargs):
    """Calculate the Risso entropy of the given prices.

    Parameters
    ----------
    prices : Dataframe
        Description of prices parameter.
    window_size : int
        Description of window_size parameter.
    **kwargs
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If 'window_size' is not valid.

    Returns
    -------
    None
    """
    if not window_size or window_size < 1:
        raise ValueError("'window_size' must be >= 1")
    if window_size < prices.shape[1]:
        raise ValueError("'window_size' must be lower than the total days")

    marks = _computeMarks(prices, **kwargs)

    entropies = []

    for i in range(prices.shape[1]):
        asset_marks = marks[:, i]

        # Sacamos todas las secuencias de window_size dias
        sequences = [
            tuple(asset_marks[i : i + window_size])
            for i in range(len(asset_marks) - window_size + 1)
        ]

        # Calculo las frecuencias que aparecen al menos una vez
        sequence_counts = {}
        for seq in sequences:
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1

        total_sequences = len(sequences)

        # N_0 cantidad de secuencias con frecuencia > 0
        N_0 = len(sequence_counts)

        # (Veces que aparece)/(Total secuencias) para sacar probabilidad de esa secuencia
        probabilities = (
            np.array(list(sequence_counts.values())) / total_sequences
        )

        # H(l) per se
        entropy = (
            -np.sum(probabilities * np.log2(probabilities)) / np.log2(N_0)
            if N_0 > 1
            else 0
        )

        entropies.append(entropy)

    return entropies


def HOne(weights):
    weights = np.asarray(weights)
    return 1 - np.sum(weights**2)


def HInf(weights):
    weights = np.asarray(weights)
    return 1 - np.max(weights)


def yagerOne(weights):
    """
    Computes Yager's entropy for a fuzzy set.

    Parameters:
    - weights: array-like, list of membership degrees (values in [0, 1])
    - p: float, parameter to control sensitivity (default = 1)

    Returns:
    - entropy: float, Yager's entropy
    """

    n = len(weights)
    weights = np.asarray(weights)

    return (sum((weights - 1 / n) ** p)) ** (1 / p)


def yagerInf(weigths):
    weigths = np.asarray(weigths)
    n = len(weigths)

    return np.max(weigths) - 1 / n
