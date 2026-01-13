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

"""Mean variance optimizers.

Implementations of different mean-variance models. There is a more general
class that can be used to apply a wider range of mean-variance models. And a
concrete class for the Markowitz model.

Key Features:
    - Portfolio optimization
    - Mean-variance models
    - Markowitz model

Examples
--------
    >>> import garpar
    >>> prices_df = [[...], [...]]  # Your price data
    >>> ss = garpar.mkss(
    ...     prices=prices_df,
    ...     weights=[0.4, 0.3, 0.3],
    ...     window_size=5
    ... )
    >>> from garpar.optimize.mean_variance import MVOptimizer
    >>> modl = MVOptimizer(model="max_sharpe", risk_free_rate=0.11)
    >>> modl.optimize(ss)

    or

    >>> from garpar import mkss
    >>> from garpar.optimize.mean_variance import Markowitz
    >>> prices_df = [[...], [...]]  # Your price data
    >>> ss = mkss(prices=prices_df, weights=[0.4, 0.3, 0.3], window_size=5)
    >>> modl = Markowitz(model="efficient_risk", target_return=0.07)
    >>> modl.optimize(ss)

See Also
--------
    PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/

"""


# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import attr

import numpy as np

import pypfopt

from .opt_base import MeanVarianceFamilyMixin, OptimizerABC
from ..constants import EPSILON
from ..utils import mabc


# =============================================================================
# METHODS
# =============================================================================


def _mv_min_volatility(instance, optimizer, ss):
    """Optimize a StocksSet to minize risk.

    Returns
    -------
    weights
        Array of weights
    metadata
        Dictionary with metadata
    """
    weights_dict = optimizer.min_volatility()
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {"name": "min_volatility"}


def _mv_max_sharpe(instance, optimizer, ss):
    """Optimize a StocksSet based on the max Sharpe's ratio.

    Returns
    -------
    weights
        Array of weights
    metadata
        Dictionary with metadata
    """
    risk_free_rate = instance._coerce_risk_free_rate(ss)

    weights_dict = optimizer.max_sharpe(risk_free_rate=risk_free_rate)
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {"name": "max_sharpe", "risk_free_rate": risk_free_rate}


def _mv_max_quadratic_utility(instance, optimizer, ss):
    """Optimize a StocksSet based on the max quadratic utility.

    Returns
    -------
    weights
        Array of weights
    metadata
        Dictionary with metadata
    """
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
    """Optimize a StocksSet based on a specific risk value.

    Returns
    -------
    weights
        Array of weights
    metadata
        Dictionary with metadata
    """
    target_volatility = instance._coerce_target_risk(ss)

    weights_dict = optimizer.efficient_risk(
        target_volatility, market_neutral=instance.market_neutral
    )
    weights = [weights_dict[stock] for stock in ss.stocks]

    return weights, {
        "name": "efficient_risk",
        "target_volatility": target_volatility,
    }


def _mv_efficient_return(instance, optimizer, ss):
    """Optimize a StocksSet based on a specific return value.

    Returns
    -------
    weights
        Array of weights
    metadata
        Dictionary with metadata
    """
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
    """Check a StocksSet best performance, do not use to optimize.

    Returns
    -------
    weights
        Array of weights
    metadata
        Dictionary with metadata
    """
    risk_free_rate = instance._coerce_risk_free_rate(ss)

    weigths_dict = optimizer.portfolio_performance(risk_free_rate)
    weights = [weigths_dict[stock] for stock in ss.stocks]

    return weights, {"name": "portfolio_performance"}


# =============================================================================
# REGISTER
# =============================================================================

MV_OPTIMIZATION_MODELS = {
    "min_volatility": _mv_min_volatility,
    "max_sharpe": _mv_max_sharpe,
    "max_quadratic_utility": _mv_max_quadratic_utility,
    "efficient_risk": _mv_efficient_risk,
    "efficient_return": _mv_efficient_return,
    "portfolio_performance": _mv_portfolio_performance,
}


# =============================================================================
# MVOptimizer
# =============================================================================


@attr.s(repr=False)
class MVOptimizer(MeanVarianceFamilyMixin, OptimizerABC):
    """Flexible Mean Variance Optimizer.

    An instance of this class represents a mean-variance optimization model.
    This class also provides different coercion methods for each model
    parameter. The parameter model indicates the optimization model to use.

    """

    model = mabc.hparam(default="max_sharpe")

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

    @model.validator
    def _check_model(self, attribute, value):
        model_func = MV_OPTIMIZATION_MODELS.get(value, value)
        if not callable(model_func):
            raise ValueError("'model' is not callable")

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

        warnings.warn("No risk_free_rate specified, coercing it")

        expected_returns = list(ss.ereturns())
        return np.median(expected_returns)

    def _coerce_risk_aversion(self, ss):
        if self.risk_aversion is not None:
            return self.risk_aversion

        warnings.warn("No risk_aversion specified, coercing it")

        expected_returns = list(ss.ereturns())
        risk_aversion = 1 / np.var(expected_returns)
        return risk_aversion

    def _coerce_target_return(self, ss):
        if self.target_return is not None:
            return self.target_return

        warnings.warn("No target_return specified, coercing it")

        returns = ss.as_returns().to_numpy().flatten()
        returns = returns[(returns != 0) & (~np.isnan(returns))]
        return np.min(np.abs(returns))

    def _coerce_target_risk(self, ss):
        if self.target_risk is not None:
            return self.target_risk

        warnings.warn("No target_risk specified, coercing it")

        cov_matrix = ss.covariance(self.covariance, **self.covariance_kw)

        return np.sqrt(1 / np.sum(np.linalg.pinv(cov_matrix))) + EPSILON

    def _calculate_weights(self, ss):
        """Calculate the optimal weights for the stocks set.

        Parameters
        ----------
        ss : garpar.core.stocks_set.StocksSet
            The stocks set to optimize.

        Returns
        -------
        tuple
            A tuple containing the optimal weights and optimizer metadata.
        """
        optimizer = self._get_optimizer(ss)
        model_func = MV_OPTIMIZATION_MODELS.get(self.model, self.model)
        return model_func(instance=self, optimizer=optimizer, ss=ss)


# =============================================================================
# MARKOWITZ
# =============================================================================


@attr.s(repr=False)
class Markowitz(MeanVarianceFamilyMixin, OptimizerABC):
    """Classic Markowitz model.

    An instance of this class represents the Markowitz optimization model with
    specific parameters. This class also provides different coercion methods
    for either the target return or target risk.
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
        ss : garpar.core.stocks_set.StocksSet
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
        ss : garpar.core.stocks_set.StocksSet
            The stocks set to optimize.

        Returns
        -------
        float
            The coerced target return.
        """
        if self.target_return:
            return self.target_return

        print("Warning: no target_return specified, coercing it")

        returns = ss.as_returns().to_numpy().flatten()
        returns = returns[returns != 0]
        returns = returns[~np.isnan(returns)]
        return np.min(np.abs(returns))

    def _coerce_target_risk(self, ss):
        """Coerce the target risk.

        Parameters
        ----------
        ss : garpar.core.stocks_set.StocksSet
            The stocks set to optimize.

        Returns
        -------
        float
            The coerced target risk.
        """
        if self.target_risk is not None:
            return self.target_risk

        print("Warning: no target_risk specified, coercing it")

        cov_matrix = ss.covariance(self.covariance, **self.covariance_kw)

        return np.sqrt(1 / np.sum(np.linalg.pinv(cov_matrix))) + EPSILON

    def _get_model_and_target_value(self, ss):
        optimizer = self._get_optimizer(ss)

        if self.target_return:
            return (
                optimizer.efficient_return,
                self._coerce_target_return(ss),
                "target_return",
            )

        return (
            optimizer.efficient_risk,
            self._coerce_target_risk(ss),
            "target_risk",
        )

    def _calculate_weights(self, ss):
        """Calculate the optimal weights for the stocks set.

        Parameters
        ----------
        ss : garpar.core.stocks_set.StocksSet
            The stocks set to optimize.

        Returns
        -------
        tuple
            A tuple containing the optimal weights and optimizer metadata.
        """
        model, target_value, value_name = self._get_model_and_target_value(ss)

        market_neutral = self._get_market_neutral()

        weights_dict = model(target_value, market_neutral=market_neutral)
        weights = [weights_dict[stock] for stock in ss.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            f"{value_name}": target_value,
        }

        return weights, optimizer_metadata
