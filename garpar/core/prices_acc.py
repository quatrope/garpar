# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Prices Accessor."""

import attr

import numpy as np

from ..utils import accabc

# =============================================================================
# STATISTIC ACCESSOR
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class PricesAccessor(accabc.AccessorABC):
    """Accessor for price-related data and methods.

    The PricesAccessor class provides a convenient interface to perform various
    statistical and mathematical operations on price data, using a predefined
    whitelist of allowable methods.

    Attributes
    ----------
    _default_kind : str
        The default kind of operation, default is "describe".
    _pf : attr.ib
        The portfolio object containing price data and other attributes.
    _DF_WHITELIST : list of str
        A list of allowable DataFrame methods.
    _GARPAR_WHITELIST : list of str
        A list of allowable custom methods.
    _WHITELIST : list of str
        A combined list of allowable methods from _DF_WHITELIST and _GARPAR_WHITELIST.

    Methods
    -------
    __getattr__(a)
        Dynamically retrieve whitelisted attributes.
    __dir__()
        List available attributes, including whitelisted methods.
    log()
        Apply the natural logarithm to the price data.
    log10()
        Apply the base 10 logarithm to the price data.
    log2()
        Apply the base 2 logarithm to the price data.
    mad(skipna=True)
        Compute the mean absolute deviation of the price data.
    """

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
        """
        Dynamically retrieve whitelisted attributes.

        Parameters
        ----------
        a : str
            The attribute name to retrieve.

        Returns
        -------
        object
            The requested attribute if it is in the whitelist.

        Raises
        ------
        AttributeError
            If the requested attribute is not in the whitelist.
        """
        if a not in self._WHITELIST:
            raise AttributeError(a)
        target = self if a in self._GARPAR_WHITELIST else self._pf._prices_df

        return getattr(target, a)

    def __dir__(self):
        """List available attributes, including whitelisted methods.

        Returns
        -------
        list of str
            A list of attribute names.
        """
        return super().__dir__() + [
            e for e in dir(self._pf._prices_df) if e in self._WHITELIST
        ]

    def log(self):
        """Apply the natural logarithm to the price data.

        Returns
        -------
        DataFrame
            The natural logarithm of the price data.
        """
        return self._pf._prices_df.apply(np.log)

    def log10(self):
        """Apply the base 10 logarithm to the price data.

        Returns
        -------
        DataFrame
            The base 10 logarithm of the price data.
        """
        return self._pf._prices_df.apply(np.log10)

    def log2(self):
        """Apply the base 2 logarithm to the price data.

        Returns
        -------
        DataFrame
            The base 2 logarithm of the price data.
        """
        return self._pf._prices_df.apply(np.log2)

    def mad(self, skipna=True):
        """Return the mean absolute deviation of the values over a given axis.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        
        Returns
        -------
        Series
            The mean absolute deviation of the values.
        """
        df = self._pf._prices_df
        return (df - df.mean(axis=0)).abs().mean(axis=0, skipna=skipna)
