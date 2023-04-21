# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import attr

import numpy as np

from ..utils import accabc

# =============================================================================
# STATISTIC ACCESSOR
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class PricesAccessor(accabc.AccessorABC):
    _default_kind = "describe"

    _pf = attr.ib()

    _DF_WHITELIST = [
        "corr",
        "cov",
        "describe",
        "kurtosis",
        "max",
        "mean",
        "median",
        "min",
        "pct_change",
        "quantile",
        "sem",
        "skew",
        "std",
        "var",
    ]

    _GARPAR_WHITELIST = ["log", "log10", "log2", "mad"]

    _WHITELIST = _DF_WHITELIST + _GARPAR_WHITELIST

    def __getattr__(self, a):
        if a not in self._WHITELIST:
            raise AttributeError(a)
        target = self if a in self._GARPAR_WHITELIST else self._pf._prices_df

        return getattr(target, a)

    def __dir__(self):
        return super().__dir__() + [
            e for e in dir(self._pf._prices_df) if e in self._WHITELIST
        ]

    def log(self):
        return self._pf._prices_df.apply(np.log)

    def log10(self):
        return self._pf._prices_df.apply(np.log10)

    def log2(self):
        return self._pf._prices_df.apply(np.log2)

    def mad(self, skipna=True):
        """Return the mean absolute deviation of the values over a given axis.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.

        """
        df = self._pf._prices_df
        return (df - df.mean(axis=0)).abs().mean(axis=0, skipna=skipna)
