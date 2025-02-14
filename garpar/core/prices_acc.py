# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Prices Accessor.

The prices accessor module offers an accessor class to perform a variety of
statistical and mathematical operations on stock prices, leveraging a
predefined whitelist of permitted methods.

Key Features:
    - Price-related data and methods

Example
-------
    >>> import garpar
    >>> ss = garpar.mkss(prices=[...])
    >>> ss.prices.describe()
    >>> ss.prices.log()

"""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np

from ..utils import AccessorABC

# =============================================================================
# STATISTIC ACCESSOR
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class PricesAccessor(AccessorABC):
    """Accessor for price-related data and methods.

    The PricesAccessor class provides a convenient interface to perform various
    statistical and mathematical operations on price data, using a predefined
    whitelist of allowable methods.
    """

    _default_kind = "describe"

    _ss = attr.ib()

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
        target = self if a in self._GARPAR_WHITELIST else self._ss._prices_df

        return getattr(target, a)

    def __dir__(self):
        """List available attributes, including whitelisted methods.

        Returns
        -------
        list of str
            A list of attribute names.
        """
        return super().__dir__() + [
            e for e in dir(self._ss._prices_df) if e in self._WHITELIST
        ]

    def log(self):
        """Apply the natural logarithm to the price data.

        Returns
        -------
        DataFrame
            The natural logarithm of the price data.
        """
        return self._ss._prices_df.apply(np.log)

    def log10(self):
        """Apply the base 10 logarithm to the price data.

        Returns
        -------
        DataFrame
            The base 10 logarithm of the price data.
        """
        return self._ss._prices_df.apply(np.log10)

    def log2(self):
        """Apply the base 2 logarithm to the price data.

        Returns
        -------
        DataFrame
            The base 2 logarithm of the price data.
        """
        return self._ss._prices_df.apply(np.log2)

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
        df = self._ss._prices_df
        return (df - df.mean(axis=0)).abs().mean(axis=0, skipna=skipna)

    def mean_tendency_size(self):
        """Compute the mean size of consecutive winning or losing streaks.

        Returns
        -------
        Series
            A Series with the mean streak size for each asset, representing
            the average length of consecutive up or down movements in returns.
        """

        def count_consecutive(stock_groups):
            # Calculate the size of each consecutive group
            stock_counts = stock_groups.groupby(stock_groups).size()

            # Return the mean size of consecutive groups
            return stock_counts.mean()

        # Convert the entire DataFrame to boolean values
        # True if return is positive, False otherwise
        wins = self._ss.as_returns() > 0

        # Detect changes in the sequence of wins/losses
        # Shift all values down by one position and compare
        # False indicates a change in the sequence
        # (win to loss or vice versa)
        changes = wins != wins.shift()

        # Use cumulative sum to assign a unique identifier to each consecutive
        # group. Each time a change is detected (False in 'changes'), the group
        # number increments
        groups = changes.cumsum()

        # Apply the count_consecutive function to each column
        # This calculates the mean size of
        # consecutive win/loss streaks
        counts = groups.apply(count_consecutive)

        # Name the resulting Series for clarity
        counts.name = "mean_tendency_size"

        return counts
