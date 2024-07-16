# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Mixins for Garpar."""

import numpy as np

import pandas as pd


class CoercerMixin:
    """A mixin class that contains utility methods for various accessors.

    This class provides methods to coerce expected returns, weights, and covariance
    matrices into desired formats.

    Attributes
    ----------
    _FORBIDDEN_METHODS : list of str
        A list of method names that are forbidden.

    Methods
    -------
    coerce_expected_returns(expected_returns, kw, asarray=True)
        Coerce expected returns into the desired format.
    coerce_weights(weights, asarray=True)
        Coerce weights into the desired format.
    coerce_covariance_matrix(cov_matrix, kw, asarray=True)
        Coerce covariance matrices into the desired format.
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

        Examples
        --------
        >>> mixin = CoercerMixin()
        >>> er = mixin.coerce_expected_returns("capm", kw={"risk_free_rate": 0.02})
        """
        if isinstance(expected_returns, str):
            kw = {} if kw is None else kw
            expected_returns = self._pf.ereturns(
                expected_returns.lower(), **kw
            )
        return np.asarray(expected_returns) if asarray else expected_returns

    def coerce_weights(self, weights, asarray=True):
        """Coerce weights into the desired format.

        Parameters
        ----------
        weights : None, Portfolio, or array-like
            The weights specification or values. If None, equal weights are assigned.
        asarray : bool, optional
            Whether to return the result as a numpy array, by default True.

        Returns
        -------
        array-like
            The coerced weights.

        Examples
        --------
        >>> mixin = CoercerMixin()
        >>> weights = mixin.coerce_weights(None)
        """
        if weights is None:
            cols = len(self._pf.stocks)
            weights = np.full(cols, 1.0 / cols, dtype=float)
        elif isinstance(weights, type(self._pf)):
            bench_weights = weights.weights

            stocks = self._pf.stocks
            # creamos un lugar donde poner los precios en el mismo orden que
            # en el pf original
            weights = pd.Series(
                np.zeros(len(stocks), dtype=float), index=stocks
            )

            for stock in stocks:
                weights[stock] = bench_weights[stock]

        return np.asarray(weights) if asarray else weights

    def coerce_covariance_matrix(self, cov_matrix, kw, asarray=True):
        """Coerce covariance matrices into the desired format.

        Parameters
        ----------
        cov_matrix : str or array-like
            The covariance matrix specification or values.
        kw : dict
            Additional keyword arguments for the covariance method.
        asarray : bool, optional
            Whether to return the result as a numpy array, by default True.

        Returns
        -------
        array-like
            The coerced covariance matrix.

        Examples
        --------
        >>> mixin = CoercerMixin()
        >>> cov = mixin.coerce_covariance_matrix("sample_cov", kw={"min_periods": 1})
        """
        if isinstance(cov_matrix, str):
            kw = {} if kw is None else kw
            cov_matrix = self._pf.covariance(cov_matrix.lower(), **kw)
        return np.asarray(cov_matrix) if asarray else cov_matrix
