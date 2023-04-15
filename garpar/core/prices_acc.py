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
        "mad",
        "max",
        "mean",
        "median",
        "info",
        "min",
        "pct_change",
        "quantile",
        "sem",
        "skew",
        "std",
        "var",
    ]

    def log(self):
        return self._pf._df.apply(np.log10)

    def __getattr__(self, a):
        if a not in self._DF_WHITELIST:
            raise AttributeError(a)
        return getattr(self._pf._df, a)

    def __dir__(self):
        return super().__dir__() + [
            e for e in dir(self._pf._df) if e in self._DF_WHITELIST
        ]
