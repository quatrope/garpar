# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Entropy measure module.

A collection of functions for calculating entropy measures.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np

import pypfopt

from scipy import stats


# =============================================================================
# ENTROPY
# =============================================================================


def _compute_marks(prices, **kwargs):
    returns = pypfopt.expected_returns.returns_from_prices(
        prices=prices, **kwargs
    )

    avg_returns = returns.mean(axis=0)

    return (returns.values >= avg_returns.values).astype(int)


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
    array-like
        The Risso entropy of the prices.
    """
    if window_size <=0 or window_size > prices.shape[0]:
        raise ValueError(
            "'window_size' must be in the interval (0, days]"
        )

    marks = _compute_marks(prices, **kwargs)

    entropies = []

    for i in range(prices.shape[1]):
        asset_marks = marks[:, i]

        sequences = []
        for idx in range(len(asset_marks) - window_size + 1):
            first, last = idx, idx + window_size
            sequence = asset_marks[first:last]
            sequences.append(tuple(sequence))

        sequence_counts = {}
        for seq in sequences:
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1

        total_sequences = len(sequences)

        N_0 = len(sequence_counts)

        probabilities = (
            np.array(list(sequence_counts.values())) / total_sequences
        )

        entropy = (
            -np.sum(probabilities * np.log2(probabilities)) / np.log2(N_0)
            if N_0 > 1
            else 0
        )

        entropies.append(entropy)

    return entropies


def yager_one(weights):
    """Compute Yager's entropy for a fuzzy set and z=1.

    Parameters
    ----------
    weights : array-like
        List of membership degrees (values in [0, 1])

    Returns
    -------
    float
        Yager's entropy for z->1
    """
    n = len(weights)
    weights = np.asarray(weights)

    return sum(abs(weights - 1 / n))


def yager_inf(weigths):
    """Compute Yager's entropy for a fuzzy set and z->inf.

    Parameters
    ----------
    weights : array-like
        List of membership degrees (values in [0, 1])

    Returns
    -------
    float
        Yager's entropy for z->inf
    """
    weigths = np.asarray(weigths)
    n = len(weigths)

    return np.max(weigths) - 1 / n
