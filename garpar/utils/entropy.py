# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

from scipy import stats

import warnings


def shannon(prices, window_size=None, **kwargs):
    if window_size is not None:
        warnings.warn(
            f"'window_size={window_size}' is ignored in shannon entropy"
        )
    return stats.entropy(prices, axis=0, **kwargs)


def risso(prices, window_size, **kwargs):
    if not window_size or window_size < 0:
        raise ValueError(f"'window_size' must be >= 0")
