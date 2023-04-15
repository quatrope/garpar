from scipy import stats

import warnings


def shannon(prices, window_size=None, **kwargs):
    if window_size is not None:
        warnings.warn(
            f"'window_size={window_size}' is ignored in shannon entropy"
        )
    return stats.entropy(prices, axis=0, **kwargs)


def risso(prices, window_size=None, **kwargs):
    if window_size is None:
        raise ValueError(f"'window_size' is required for risso entropy")



