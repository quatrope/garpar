import attr

import numpy as np

from ..utils import aabc

# =============================================================================
# STATISTIC ACCESSOR
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class PricesAccessor(aabc.AccessorABC):

    _DEFAULT_KIND = "describe"

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
