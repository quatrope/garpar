import attr

# =============================================================================
# STATISTIC ACCESSOR
# =============================================================================


@attr.s(repr=False, cmp=False)
class PricesAccessor:

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

    def __call__(self, statistic="describe", **kwargs):
        if statistic.startswith("_"):
            raise ValueError(f"invalid statistic name '{statistic}'")
        method = getattr(self, statistic, None)
        if not callable(method):
            raise ValueError(f"invalid statistic name '{statistic}'")
        return method(**kwargs)

    def __getattr__(self, a):
        if a not in self._DF_WHITELIST:
            raise AttributeError(a)
        return getattr(self._pf._df, a)

    def __dir__(self):
        return [e for e in dir(self._pf._df) if e in self._DF_WHITELIST]
