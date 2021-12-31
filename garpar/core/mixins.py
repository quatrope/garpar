import numpy as np


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
            # validar que sean los mismos activos
            weights = weights._weights
        return np.asarray(weights) if asarray else weights

    def coerce_covariance_matrix(self, cov_matrix, kw, asarray=True):
        if isinstance(cov_matrix, str):
            kw = {} if kw is None else kw
            cov_matrix = self._pf.covariance(cov_matrix.lower(), **kw)
        return np.asarray(cov_matrix) if asarray else cov_matrix
