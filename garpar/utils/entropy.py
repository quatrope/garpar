# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Entropy."""

from scipy import stats

import warnings


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


def risso(prices, window_size, **kwargs):
    """Calculate the Risso entropy of the given prices.

    Parameters
    ----------
    prices : array_like
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
    if not window_size or window_size < 0:
        raise ValueError(f"'window_size' must be >= 0")
