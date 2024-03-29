import attr

from pypfopt import objective_functions

from . import _mixins
from ..utils import accabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class UtilitiesAccessor(accabc.AccessorABC, _mixins.CoercerMixin):
    _default_kind = "pf_return"

    _pf = attr.ib()

    def ex_ante_tracking_error(
        self,
        *,
        covariance="sample_cov",
        covariance_kw=None,
        benchmark_weights=None,
    ):
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)
        benchmark_weights = self.coerce_weights(benchmark_weights)

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
        historic_returns = self.coerce_expected_returns(
            historic_returns, historic_returns_kw
        )
        benchmark_returns = self.coerce_expected_returns(
            benchmark_returns, benchmark_returns_kw
        )
        return objective_functions.ex_post_tracking_error(
            self._pf._weights,
            historic_returns=historic_returns,
            benchmark_returns=benchmark_returns,
        )

    def portfolio_return(
        self,
        *,
        expected_returns="capm",
        expected_returns_kw=None,
        negative=True,
    ):
        expected_returns = self.coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        return objective_functions.portfolio_return(
            self._pf._weights,
            expected_returns=expected_returns,
            negative=negative,
        )

    pf_return = portfolio_return

    def quadratic_utility(
        self,
        *,
        expected_returns="capm",
        covariance="sample_cov",
        risk_aversion=0.5,
        expected_returns_kw=None,
        covariance_kw=None,
        **kwargs,
    ):
        expected_returns = self.coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)

        return objective_functions.quadratic_utility(
            self._pf._weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_aversion=risk_aversion,
            **kwargs,
        )

    qutility = quadratic_utility
