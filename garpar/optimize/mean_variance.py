# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Mean variance optimizers."""

import attr

from .base import OptimizerABC, MeanVarianceFamilyMixin

from ..utils import mabc

import numpy as np

import pypfopt

# =============================================================================
# OPTIMIZER
# =============================================================================


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
    risk_free_rate = mabc.hparam(default=0.02)

    def _get_optimizer(self, pf):
        expected_returns = pf.ereturns(self.returns, **self.returns_kw)
        cov_matrix = pf.covariance(self.covariance, **self.covariance_kw)
        return pypfopt.EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            weight_bounds=self.weight_bounds,
        )

    def _coerce_risk_aversion(self, pf):
        return 1 # FIXME Implementar

    def _coerce_target_return(self, pf):
        if self.target_return is None:
            returns = pf.as_returns().to_numpy().flatten()
            returns = returns[(returns != 0) & (~np.isnan(returns))]
            return np.min(np.abs(returns))
        return self.target_return

    def _coerce_target_volatility(self, pf):
        if self.target_risk is None:
            return np.min(np.std(pf.as_prices()))
        return self.target_risk

    def _calculate_weights(self, pf):
        optimizer = self._get_optimizer(pf)
        method = self.method
        
        optimization_methods = {
            "min_volatility": self._min_volatility,
            "max_sharpe": self._max_sharpe,
            "max_quadratic_utility": self._max_quadratic_utility,
            "efficient_risk": self._efficient_risk,
            "efficient_return": self._efficient_return,
            "portfolio_performance": self._portfolio_performance,
        }
        
        if method not in optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return optimization_methods[method](optimizer, pf)

    def _min_volatiliy(self, optimizer, pf):
        weights_dict = optimizer.min_volatility()
        weights = [weights_dict[stock] for stock in pf.stocks]
        return weights, {"name": "min_volatility"}

    def _max_sharpe(self, optimizer, pf):
        weights_dict = optimizer.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = [weights_dict[stock] for stock in pf.stocks]
        return weights, {"name": "max_sharpe", "risk_free_rate": self.risk_free_rate}

    def _max_quadratic_utility(self, optimizer, pf):
        risk_aversion = self._coerce_risk_aversion(pf)
        weights_dict = optimizer.max_quadratic_utility(
            risk_aversion, market_neutral=self.market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]
        return weights, {"name": "max_quadratic_utility", "risk_aversion": risk_aversion}

    def _efficient_risk(self, optimizer, pf):
        target_volatility = self._coerce_target_volatility(pf)
        weights_dict = optimizer.efficient_risk(
            target_volatility, market_neutral=self.market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]
        return weights, {"name": "efficient_risk", "target_volatility": target_volatility}

    def _efficient_return(self, optimizer, pf):
        target_return = self._coerce_target_return(pf)
        weights_dict = optimizer.efficient_return(
            target_return, market_neutral=self.market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]
        return weights, {"name": "efficient_return", "target_return": target_return}

    def _portfolio_performance(self, optimizer, pf):
        weigths_dict = optimizer.portfolio_performance(self.risk_free_rate)
        weights = [weigths_dict[stock] for stock in pf.stocks]
        return weights, {"name": "portfolio_performance"}


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

    method = mabc.hparam(default="max_sharpe")

    weight_bounds = mabc.hparam(default=(0, 1))
    market_neutral = mabc.hparam(default=False)

    returns = mabc.hparam(default="mah")
    returns_kw = mabc.hparam(factory=dict)

    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)

    def _get_optimizer(self, pf):
        """Get the pypfopt EfficientFrontier optimizer.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        pypfopt.EfficientFrontier
            The configured optimizer.
        """
        expected_returns = pf.ereturns(self.returns, **self.returns_kw)
        cov_matrix = pf.covariance(self.covariance, **self.covariance_kw)
        weight_bounds = self.weight_bounds
        optimizer = pypfopt.EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            weight_bounds=weight_bounds,
        )
        return optimizer
    
    def _get_market_neutral(self):
        return self.market_neutral

    def _coerce_target_return(self, pf):
        """Coerce the target return.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        float
            The coerced target return.
        """
        if self.target_return is None:
            returns = pf.as_returns().to_numpy().flatten()
            returns = returns[returns != 0]
            returns = returns[~np.isnan(returns)]
            return np.min(np.abs(returns))
        return self.target_return

    def _calculate_weights(self, pf):
        """Calculate the optimal weights for the portfolio.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        tuple
            A tuple containing the optimal weights and optimizer metadata.
        """
        optimizer = self._get_optimizer(pf)
        target_return = self._coerce_target_return(pf)
        market_neutral = self._get_market_neutral()

        weights_dict = optimizer.efficient_return(
            target_return, market_neutral=market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            "target_return": target_return,
        }

        return weights, optimizer_metadata