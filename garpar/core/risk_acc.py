import attr

import numpy as np

from pypfopt import expected_returns

from ..utils import aabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class RiskAccessor(aabc.AccessorABC):

    _DEFAULT_KIND = "beta"

    _pf = attr.ib()

    def _coerce_weights(self, weights):
        if weights is None:
            cols = len(self._pf.stocks)
            weights = np.full(cols, 1.0 / cols, dtype=float)
        elif isinstance(weights, type(self._pf)):
            # validar que sean los mismos activos
            weights = weights._weights
        return np.asarray(weights)

    def _returns_df(self, market_prices, log_returns):
        prices = self._pf._df
        market_returns = None

        returns = expected_returns.returns_from_prices(prices, log_returns)
        if market_prices is not None:
            market_returns = expected_returns.returns_from_prices(
                market_prices, log_returns
            )

        mkt_col, idx = "_mkt_", 0
        while mkt_col in returns.columns:
            mkt_col = f"_mkt_{idx}_"
            idx += 1

        # Use the equally-weighted dataset as a proxy for the market
        if market_returns is None:
            # Append market return to right and compute sample covariance matrix
            returns[mkt_col] = returns.mean(axis=1)
        else:
            market_returns.name = mkt_col
            returns = returns.join(market_returns, how="left")
        return returns, mkt_col

    def stock_beta(
        self,
        market_prices=None,
        log_returns=False,
    ):
        returns, mkt_col = self._returns_df(market_prices, log_returns)

        # Compute covariance matrix for the new dataframe (including markets)
        cov = returns.cov()
        # The far-right column of the cov matrix is covariances to market
        betas = cov[mkt_col] / cov.loc[mkt_col, mkt_col]
        betas = betas.drop(mkt_col)
        betas.name = "beta"

        return betas

    def pf_beta(self, *, benchmark_weights=None, log_returns=False):

        benchmark_weights = self._coerce_weights(benchmark_weights)

        day_weighted_prices = np.sum(self._pf._df * self._pf._weights, axis=1)
        returns, mkt_col = self._returns_df(
            market_prices=day_weighted_prices, log_returns=log_returns
        )

        benchmark_day_weighted_prices = np.sum(
            self._pf._df * benchmark_weights, axis=1
        )
        benchmark_returns, benchmark_mkt_col = self._returns_df(
            market_prices=benchmark_day_weighted_prices,
            log_returns=log_returns,
        )

        return_cov = returns.cov().loc[mkt_col, mkt_col]
        benchmark_cov = benchmark_returns.cov().loc[
            benchmark_mkt_col, benchmark_mkt_col
        ]

        return return_cov / benchmark_cov

    def treynor_ratio(
        self,
        *,
        expected_returns="capm",
        expected_returns_kw=None,
        negative=True,
        benchmark_weights=None,
        log_returns=False,
    ):
        pf_return = self._pf.utilities.pf_return(
            expected_returns=expected_returns,
            expected_returns_kw=expected_returns_kw,
            negative=negative,
        )
        pf_beta = self.pf_beta(
            benchmark_weights=benchmark_weights, log_returns=log_returns
        )
        return pf_return / pf_beta
