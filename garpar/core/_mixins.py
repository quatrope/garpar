# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import numpy as np

import pandas as pd


class CoercerMixin:
    """Esta clase contiene utilidades funcionales a varios accersors"""

    _FORBIDDEN_METHODS = [
        "coerce_covariance_matrix",
        "coerce_expected_returns",
        "coerce_weights",
    ]

    def coerce_expected_returns(self, expected_returns, kw, asarray=True):
        if isinstance(expected_returns, str):
            kw = {} if kw is None else kw
            expected_returns = self._pf.ereturns(
                expected_returns.lower(), **kw
            )
        return np.asarray(expected_returns) if asarray else expected_returns

    def coerce_weights(self, weights, asarray=True):
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
        if isinstance(cov_matrix, str):
            kw = {} if kw is None else kw
            cov_matrix = self._pf.covariance(cov_matrix.lower(), **kw)
        return np.asarray(cov_matrix) if asarray else cov_matrix
