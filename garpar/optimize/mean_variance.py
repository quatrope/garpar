# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Mean variance optimizers."""

import attr

from .opt_base import OptimizerABC, MeanVarianceFamilyMixin

from ..utils import mabc

import numpy as np

import pypfopt

# =============================================================================
# MVOptimizer
# =============================================================================

# METHODS =====================================================================


def _mv_min_volatility(instance, optimizer, ss):
    weights_dict = optimizer.min_volatility()
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {"name": "min_volatility"}


def _mv_max_sharpe(instance, optimizer, ss):
    risk_free_rate = instance._coerce_risk_free_rate(ss)

    weights_dict = optimizer.max_sharpe(risk_free_rate=risk_free_rate)
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {"name": "max_sharpe", "risk_free_rate": risk_free_rate}


def _mv_max_quadratic_utility(instance, optimizer, ss):
    risk_aversion = instance._coerce_risk_aversion(ss)

    weights_dict = optimizer.max_quadratic_utility(
        risk_aversion, market_neutral=instance.market_neutral
    )
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {
        "name": "max_quadratic_utility",
        "risk_aversion": risk_aversion,
    }


def _mv_efficient_risk(instance, optimizer, ss):
    target_volatility = instance._coerce_target_volatility(ss)

    weights_dict = optimizer.efficient_risk(
        target_volatility, market_neutral=instance.market_neutral
    )
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {
        "name": "efficient_risk",
        "target_volatility": target_volatility,
    }


def _mv_efficient_return(instance, optimizer, ss):
    target_return = instance._coerce_target_return(ss)

    weights_dict = optimizer.efficient_return(
        target_return, market_neutral=instance.market_neutral
    )
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {
        "name": "efficient_return",
        "target_return": target_return,
    }


def _mv_portfolio_performance(instance, optimizer, ss):
    risk_free_rate = instance._coerce_risk_free_rate(ss)

    weigths_dict = optimizer.portfolio_performance(risk_free_rate)
    weights = [weigths_dict[stock] for stock in ss.stocks]

    return weights, {"name": "portfolio_performance"}


# REGISTER ====================================================================

MV_OPTIMIZATION_METHODS = {
    "min_volatility": _mv_min_volatility,
    "max_sharpe": _mv_max_sharpe,
    "max_quadratic_utility": _mv_max_quadratic_utility,
    "efficient_risk": _mv_efficient_risk,
    "efficient_return": _mv_efficient_return,
    "portfolio_performance": _mv_portfolio_performance,
}


@attr.define(repr=False)
class MVOptimizer(MeanVarianceFamilyMixin, OptimizerABC):
    """Flexible Mean Variance Optimizer."""

    method = mabc.hparam(default="max_sharpe")

    weight_bounds = mabc.hparam(default=(0, 1))
    market_neutral = mabc.hparam(default=False)

    returns = mabc.hparam(default="mah")
    returns_kw = mabc.hparam(factory=dict)

    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)

    target_return = mabc.hparam(default=None)
    target_risk = mabc.hparam(default=None)
    risk_free_rate = mabc.hparam(default=None)
    risk_aversion = mabc.hparam(default=None)

    @method.validator
    def _check_method(self, attribute, value):
        method_func = MV_OPTIMIZATION_METHODS.get(value, value)
        if not callable(method_func):
            raise ValueError("'method' doesn't look like a method")

    def _get_optimizer(self, ss):
        expected_returns = ss.ereturns(self.returns, **self.returns_kw)
        cov_matrix = ss.covariance(self.covariance, **self.covariance_kw)
        return pypfopt.EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            weight_bounds=self.weight_bounds,
        )

    def _coerce_risk_free_rate(self, ss):
        if self.risk_free_rate is not None:
            return self.risk_free_rate

        expected_returns = list(ss.ereturns())
        return np.median(expected_returns)

    def _coerce_risk_aversion(self, ss):
        if self.risk_aversion is not None:
            return self.risk_aversion

        expected_returns = list(ss.ereturns())
        risk_aversion = 1 / np.var(expected_returns)
        return risk_aversion

    def _coerce_target_return(self, ss):
        if self.target_return is not None:
            return self.target_return

        returns = ss.as_returns().to_numpy().flatten()
        returns = returns[(returns != 0) & (~np.isnan(returns))]
        return np.min(np.abs(returns))

    def _coerce_target_volatility(self, ss):
        if self.target_risk is not None:
            return self.target_risk

        return np.min(np.std(ss.as_prices(), axis=0))

    def _calculate_weights(self, ss):
        optimizer = self._get_optimizer(ss)
        method_func = MV_OPTIMIZATION_METHODS.get(self.method, self.method)
        return method_func(instance=self, optimizer=optimizer, ss=ss)


# =============================================================================
# MARKOWITZ
# =============================================================================
@attr.define(repr=False)
class Markowitz(MeanVarianceFamilyMixin, OptimizerABC):
    """Classic Markowitz model.

    This method implements the Classic Model Markowitz 1952 in Mansini, R.,
    WLodzimierz, O., and Speranza, M. G. (2015). Linear and mixed
    integer programming for portfolio optimization. Springer and EURO: The
    Association of European Operational Research Societies
    """

    target_return = mabc.hparam(default=None)
    target_risk = mabc.hparam(default=None)

    weight_bounds = mabc.hparam(default=(0, 1))
    market_neutral = mabc.hparam(default=False)

    returns = mabc.hparam(default="mah")
    returns_kw = mabc.hparam(factory=dict)

    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)

    def _get_optimizer(self, ss):
        """Get the pypfopt EfficientFrontier optimizer.

        Parameters
        ----------
        ss : StocksSet
            The stocks set to optimize.

        Returns
        -------
        pypfopt.EfficientFrontier
            The configured optimizer.
        """
        expected_returns = ss.ereturns(self.returns, **self.returns_kw)
        cov_matrix = ss.covariance(self.covariance, **self.covariance_kw)
        weight_bounds = self.weight_bounds
        return pypfopt.EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            weight_bounds=weight_bounds,
        )

    def _get_market_neutral(self):
        return self.market_neutral

    def _coerce_target_return(self, ss):
        """Coerce the target return.

        Parameters
        ----------
        ss : StocksSet
            The stocks set to optimize.

        Returns
        -------
        float
            The coerced target return.
        """
        if self.target_return is None:
            returns = ss.as_returns().to_numpy().flatten()
            returns = returns[returns != 0]
            returns = returns[~np.isnan(returns)]
            return np.min(np.abs(returns))
        return self.target_return

    def _calculate_weights(self, ss):
        """Calculate the optimal weights for the stocks set.

        Parameters
        ----------
        ss : StocksSet
            The stocks set to optimize.

        Returns
        -------
        tuple
            A tuple containing the optimal weights and optimizer metadata.
        """
        optimizer = self._get_optimizer(ss)
        target_return = self._coerce_target_return(ss)
        market_neutral = self._get_market_neutral()

        weights_dict = optimizer.efficient_return(
            target_return, market_neutral=market_neutral
        )
        weights = [weights_dict[stock] for stock in ss.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            "target_return": target_return,
        }

        return weights, optimizer_metadata
