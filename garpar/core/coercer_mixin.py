# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2026
#   Lautaro Ebner,
#   Diego Gimenez,
#   Nadia Luczywo,
#   Juan Cabral,
#   and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Coercer mixin for Garpar project.

The coercer mixin module provides a specific mixin class used to coerce
expected returns, weights, and covariance matrices into the desired formats.

Example
-------
    >>> from garpar import mkss
    >>> ss = mkss([...])
    >>> ss.covariance.sample_cov()
    >>> ss.covariance.ledoit_wolf_cov()
    >>> ss.covariance.oracle_approximating_cov()
    >>> ss.correlation.sample_corr()
    >>> ss.correlation.oracle_approximating_corr()

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

# =============================================================================
# COERCER MIXIN
# =============================================================================


class CoercerMixin:
    """A mixin class that contains utility methods for various accessors.

    This class provides methods to coerce expected returns, weights, and
    covariance matrices into desired formats.
    """

    _FORBIDDEN_METHODS = [
        "coerce_covariance_matrix",
        "coerce_expected_returns",
        "coerce_weights",
    ]

    def coerce_expected_returns(self, expected_returns, kw, asarray=True):
        """Coerce expected returns into the desired format.

        Parameters
        ----------
        expected_returns : str or array-like
            The expected returns specification or values.
        kw : dict
            Additional keyword arguments for the expected returns method.
        asarray : bool, optional
            Whether to return the result as a numpy array, by default True.

        Returns
        -------
        array-like
            The coerced expected returns.
        """
        if isinstance(expected_returns, str):
            kw = {} if kw is None else kw
            expected_returns = self._ss.ereturns(
                expected_returns.lower(), **kw
            )
        return np.asarray(expected_returns) if asarray else expected_returns

    def coerce_weights(self, weights, asarray=True):
        """Coerce weights into the desired format.

        Parameters
        ----------
        weights : None, StocksSet, or array-like
            The weights specification or values. If None, equal weights are
            assigned.
        asarray : bool, optional
            Whether to return the result as a numpy array, by default True.

        Returns
        -------
        array-like
            The coerced weights.
        """
        if weights is None:
            cols = len(self._ss.stocks)
            weights = np.full(cols, 1.0 / cols, dtype=float)
        elif isinstance(weights, type(self._ss)):
            bench_weights = weights.weights

            stocks = self._ss.stocks
            weights = pd.Series(
                np.zeros(len(stocks), dtype=float), index=stocks
            )

            for stock in stocks:
                weights[stock] = bench_weights[stock]

        return np.asarray(weights) if asarray else weights

    def coerce_covariance_matrix(self, cov_matrix, kw, asarray=True):
        """Coerce covariance matrix into the desired format.

        Parameters
        ----------
        cov_matrix : None, str, or array-like
            The covariance matrix specification or values. If None, the sample
            covariance matrix is used.
        kw : dict, optional
            Additional keyword arguments for the covariance matrix method,
            by default None.
        asarray : bool, optional
            Whether to return the result as a numpy array, by default True.

        Returns
        -------
        array-like
            The coerced covariance matrix.
        """
        if isinstance(cov_matrix, str):
            kw = {} if kw is None else kw
            cov_matrix = self._ss.covariance(cov_matrix.lower(), **kw)
        return np.asarray(cov_matrix) if asarray else cov_matrix
