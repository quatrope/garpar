# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Risk Accessor."""

import attr

import numpy as np

from pypfopt import expected_returns, objective_functions

from . import _mixins
from ..utils import accabc

# =============================================================================
# RISK ACCESSOR
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class RiskAccessor(accabc.AccessorABC, _mixins.CoercerMixin):
    """Accessor for various risk metrics.

    The RiskAccessor class provides methods to compute stock and stocks set
    betas, Treynor ratio, stocks set variance, Sharpe ratio, and Value at
    Risk (VaR).

    Attributes
    ----------
    _default_kind : str
        The default kind of risk measure, default is "ss_beta".
    _ss : attr.ib
        The StocksSet object containing weights, prices, and other attributes.

    Methods
    -------
    stock_beta(market_prices=None, log_returns=False)
        Computes the beta of individual stocks in the stocks_set.
    stocks_set_beta(benchmark_weights=None, log_returns=False)
        Computes the beta of the entire stocks_set.
    treynor_ratio(expected_returns='capm', expected_returns_kw=None,
                  negative=True, benchmark_weights=None, log_returns=False)
        Computes the Treynor ratio of the stocks_set.
    stocks_set_variance(covariance='sample_cov', covariance_kw=None)
        Computes the variance of the stocks_set.
    sharpe_ratio(expected_returns='capm', covariance='sample_cov',
                 expected_returns_kw=None, covariance_kw=None, **kwargs)
        Computes the Sharpe ratio of the stocks_set.
    value_at_risk(alpha=0.05)
        Computes the Value at Risk (VaR) of the stocks_set.
    """

    _default_kind = "ss_beta"

    _ss = attr.ib()

    def _returns_df(self, market_prices, log_returns):
        """Prepare the returns DataFrame.

        Parameters
        ----------
        market_prices : DataFrame
            Market prices data.
        log_returns : bool
            Whether to compute log returns.

        Returns
        -------
        tuple
            A tuple containing the returns DataFrame and the market
            column name.
        """
        prices = self._ss._prices_df
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
        """Compute the beta of individual stocks in the stocks set.

        Parameters
        ----------
        market_prices : DataFrame, optional
            Market prices data, by default None.
        log_returns : bool, optional
            Whether to compute log returns, by default False.

        Returns
        -------
        Series
            The beta values for individual stocks.

        Examples
        --------
        >>> accessor = RiskAccessor(ss)
        >>> betas = accessor.stock_beta()
        """
        returns, mkt_col = self._returns_df(market_prices, log_returns)

        # Compute covariance matrix for the new dataframe (including markets)
        cov = returns.cov()
        # The far-right column of the cov matrix is covariances to market
        betas = cov[mkt_col] / cov.loc[mkt_col, mkt_col]
        betas = betas.drop(mkt_col)
        betas.name = "beta"

        return betas

    def stocks_set_beta(self, *, benchmark_weights=None, log_returns=False):
        """Compute the beta of the entire stocks set.

        Parameters
        ----------
        benchmark_weights : array-like, optional
            The weights of the benchmark stocks set, by default None.
        log_returns : bool, optional
            Whether to compute log returns, by default False.

        Returns
        -------
        float
            The beta of the stocks set.

        Examples
        --------
        >>> accessor = RiskAccessor(ss)
        >>> beta = accessor.stocks_set_beta()
        """
        benchmark_weights = self.coerce_weights(benchmark_weights)

        day_weighted_prices = np.sum(
            self._ss._prices_df * self._ss._weights, axis=1
        )
        returns, mkt_col = self._returns_df(
            market_prices=day_weighted_prices, log_returns=log_returns
        )

        benchmark_day_weighted_prices = np.sum(
            self._ss._prices_df * benchmark_weights, axis=1
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

    ss_beta = stocks_set_beta

    def treynor_ratio(
        self,
        *,
        expected_returns="capm",
        expected_returns_kw=None,
        negative=True,
        benchmark_weights=None,
        log_returns=False,
    ):
        """Compute the Treynor ratio of the stocks set.

        Parameters
        ----------
        expected_returns : str, optional
            The method to compute the expected returns, by default 'capm'.
        expected_returns_kw : dict, optional
            Additional keyword arguments for the expected returns method,
            by default None.
        negative : bool, optional
            Whether to return the negative of the Treynor ratio,
            by default True.
        benchmark_weights : array-like, optional
            The weights of the benchmark stocks set, by default None.
        log_returns : bool, optional
            Whether to compute log returns, by default False.

        Returns
        -------
        float
            The Treynor ratio of the stocks set.

        Examples
        --------
        >>> accessor = RiskAccessor(ss)
        >>> ratio = accessor.treynor_ratio()
        """
        ss_return = self._ss.utilities.ss_return(
            expected_returns=expected_returns,
            expected_returns_kw=expected_returns_kw,
            negative=negative,
        )
        ss_beta = self.ss_beta(
            benchmark_weights=benchmark_weights, log_returns=log_returns
        )
        return ss_return / ss_beta

    def stocks_set_variance(self, covariance="sample_cov", covariance_kw=None):
        """
        Compute the variance of the stocks set.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix,
            by default 'sample_cov'.
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method,
            by default None.

        Returns
        -------
        float
            The variance of the stocks set.

        Examples
        --------
        >>> accessor = RiskAccessor(ss)
        >>> var = accessor.stocks_set_variance()
        """
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)
        return objective_functions.portfolio_variance(
            self._ss._weights, cov_matrix=cov_matrix
        )

    ss_var = stocks_set_variance

    def sharpe_ratio(
        self,
        *,
        expected_returns="capm",
        covariance="sample_cov",
        expected_returns_kw=None,
        covariance_kw=None,
        **kwargs,
    ):
        """
        Compute the Sharpe ratio of the stocks set.

        Parameters
        ----------
        expected_returns : str, optional
            The method to compute the expected returns,
            by default 'capm'.
        covariance : str, optional
            The method to compute the covariance matrix,
            by default 'sample_cov'.
        expected_returns_kw : dict, optional
            Additional keyword arguments for the expected returns method,
            by default None.
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method,
            by default None.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            The Sharpe ratio of the stocks set.

        Examples
        --------
        >>> accessor = RiskAccessor(ss)
        >>> ratio = accessor.sharpe_ratio()
        """
        expected_returns = self.coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)
        return objective_functions.sharpe_ratio(
            self._ss._weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            **kwargs,
        )

    # VaR =====================================================================
    def _stock_returns_VaR(self, stock_returns, alpha):
        """
        Compute the Value at Risk (VaR) for stock returns.

        Parameters
        ----------
        stock_returns : array-like
            The returns of the stock.
        alpha : float
            The significance level.

        Returns
        -------
        float
            The Value at Risk (VaR) for the given stock returns.
        """
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
        """
        Compute the Value at Risk (VaR) of the stocks set.

        Parameters
        ----------
        alpha : float, optional
            The significance level, by default 0.05.

        Returns
        -------
        Series
            The Value at Risk (VaR) for each stock in the stocks set.

        Examples
        --------
        >>> accessor = RiskAccessor(ss)
        >>> var = accessor.value_at_risk(alpha=0.05)
        """
        returns = self._ss.as_returns()

        var = returns.apply(self._stock_returns_VaR, axis="rows", alpha=alpha)
        var.name = "VaR"

        return var

    var = value_at_risk
