import attr

import numpy as np

from pypfopt import objective_functions

from ..utils import aabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, repr=False, slots=True)
class ObjectivesAccessor(aabc.AccessorABC):

    _DEFAULT_KIND = "pf_return"

    _pf = attr.ib()

    def _coerce_expected_returns(self, expected_returns, kw):
        if isinstance(expected_returns, str):
            kw = {} if kw is None else kw
            expected_returns = self._pf.ereturns(
                expected_returns.lower(), **kw
            )
        return np.asarray(expected_returns)

    def _coerce_covariance_matrix(self, cov_matrix, kw):
        if isinstance(cov_matrix, str):
            kw = {} if kw is None else kw
            cov_matrix = self._pf.covariance(cov_matrix.lower(), **kw)
        return np.asarray(cov_matrix)

    def _coerce_weights(self, weights):
        if weights is None:
            cols = len(self._pf.stocks)
            weights = np.full(cols, 1.0 / cols, dtype=float)
        elif isinstance(weights, type(self._pf)):
            weights = weights._weights
        return np.asarray(weights)

    def ex_ante_tracking_error(
        self,
        *,
        covariance="sample",
        covariance_kw=None,
        benchmark_weights=None,
    ):
        cov_matrix = self._coerce_covariance_matrix(covariance, covariance_kw)
        benchmark_weights = self._coerce_weights(benchmark_weights)

        return objective_functions.ex_ante_tracking_error(
            self._pf._weights,
            cov_matrix=cov_matrix,
            benchmark_weights=benchmark_weights,
        )

    def ex_post_tracking_error(
        self,
        *,
        historic_returns="capm",
        benchmark_returns="capm",
        historic_returns_kw=None,
        benchmark_returns_kw=None,
    ):
        historic_returns = self._coerce_expected_returns(
            historic_returns, historic_returns_kw
        )
        benchmark_returns = self._coerce_expected_returns(
            benchmark_returns, benchmark_returns_kw
        )
        return objective_functions.ex_post_tracking_error(
            self._pf._weights,
            historic_returns=historic_returns,
            benchmark_returns=benchmark_returns,
        )

    def L2_reg(self, **kwargs):
        return objective_functions.L2_reg(self._pf._weights, **kwargs)

    def pf_return(
        self, *, expected_returns="capm", expected_returns_kw=None, **kwargs
    ):
        expected_returns = self._coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        return objective_functions.portfolio_return(
            self._pf._weights, expected_returns=expected_returns, **kwargs
        )

    def pf_variance(self, covariance="sample", covariance_kw=None, **kwargs):
        cov_matrix = self._coerce_covariance_matrix(covariance, covariance_kw)
        return objective_functions.portfolio_variance(
            self._pf._weights, cov_matrix=cov_matrix, **kwargs
        )

    def quadratic_utility(
        self,
        *,
        expected_returns="capm",
        covariance="sample",
        risk_aversion=0.5,
        expected_returns_kw=None,
        covariance_kw=None,
        **kwargs,
    ):
        expected_returns = self._coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        cov_matrix = self._coerce_covariance_matrix(covariance, covariance_kw)

        return objective_functions.quadratic_utility(
            self._pf._weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_aversion=risk_aversion,
            **kwargs,
        )

    def sharpe(
        self,
        *,
        expected_returns="capm",
        covariance="sample",
        expected_returns_kw=None,
        covariance_kw=None,
        **kwargs,
    ):
        expected_returns = self._coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        cov_matrix = self._coerce_covariance_matrix(covariance, covariance_kw)
        return objective_functions.sharpe_ratio(
            self._pf._weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            **kwargs,
        )

    def transaction_cost(self, *, w_prev=None, **kwargs):
        w_prev = self._coerce_weights(w_prev)

        return objective_functions.transaction_cost(
            self._pf._weights, w_prev=w_prev, **kwargs
        )
