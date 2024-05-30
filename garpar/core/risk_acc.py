# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import attr

import numpy as np

from pypfopt import expected_returns, objective_functions

from . import _mixins
from ..utils import accabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class RiskAccessor(accabc.AccessorABC, _mixins.CoercerMixin):
    _default_kind = "pf_beta"

    _pf = attr.ib()

    def _returns_df(self, market_prices, log_returns):
        prices = self._pf._prices_df
        market_returns = None

        returns = expected_returns.returns_from_prices(prices, log_returns)
        if market_prices is not None:
            market_returns = expected_returns.returns_from_prices(
                market_prices, log_returns
            )

        # we ensure that the market column we are going to insert is unique
        # and does not step on a stock
        mkt_col, idx = "_mkt_", 0
        while mkt_col in returns.columns:
            mkt_col = f"_mkt_{idx}_"
            idx += 1

        # Use the equally-weighted dataset as a proxy for the market
        if market_returns is None:
            # Append market return to right and
            # compute sample covariance matrix
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

    def portfolio_beta(self, *, benchmark_weights=None, log_returns=False):
        benchmark_weights = self.coerce_weights(benchmark_weights)

        day_weighted_prices = np.sum(
            self._pf._prices_df * self._pf._weights, axis=1
        )
        returns, mkt_col = self._returns_df(
            market_prices=day_weighted_prices, log_returns=log_returns
        )

        benchmark_day_weighted_prices = np.sum(
            self._pf._prices_df * benchmark_weights, axis=1
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

    pf_beta = portfolio_beta

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

    def portfolio_variance(self, covariance="sample_cov", covariance_kw=None):
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)
        return objective_functions.portfolio_variance(
            self._pf._weights, cov_matrix=cov_matrix
        )

    pf_var = portfolio_variance

    def sharpe_ratio(
        self,
        *,
        expected_returns="capm",
        covariance="sample_cov",
        expected_returns_kw=None,
        covariance_kw=None,
        **kwargs,
    ):
        expected_returns = self.coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)
        return objective_functions.sharpe_ratio(
            self._pf._weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            **kwargs,
        )

    # VaR =====================================================================
    def _stock_returns_VaR(self, stock_returns, alpha):
        stock_returns_arr = np.array(stock_returns, ndmin=2)
        if stock_returns_arr.shape[0] == 1 and stock_returns_arr.shape[1] > 1:
            stock_returns_arr = stock_returns_arr.T
        elif stock_returns_arr.shape[0] > 1 and stock_returns_arr.shape[1] > 1:
            raise ValueError("returns must have Tx1 size")

        sorted_stock_returns_arr = np.sort(stock_returns_arr, axis=0)
        index = int(np.ceil(alpha * len(sorted_stock_returns_arr)) - 1)
        value = np.asarray(-sorted_stock_returns_arr[index]).item()

        return value

    def value_at_risk(self, *, alpha=0.05):
        returns = self._pf.as_returns()

        var = returns.apply(self._stock_returns_VaR, axis="rows", alpha=alpha)
        var.name = "VaR"

        return var

    var = value_at_risk
