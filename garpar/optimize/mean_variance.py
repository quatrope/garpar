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
    """Mean Variance Optimizer."""

    target_return = mabc.hparam(default=None)
    target_risk = mabc.hparam(default=None)

    method = mabc.hparam(default="max_sharpe")

    weight_bounds = mabc.hparam(default=(0, 1))
    market_neutral = mabc.hparam(default=False)

    returns = mabc.hparam(default="mah")
    returns_kw = mabc.hparam(factory=dict)

    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)

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

        This function checks if the target return is None. If it is, it
        calculates the target return by finding the minimum absolute value of
        the non-zero and non-NaN returns of the portfolio. Otherwise, it
        returns the target return as it is.
        """
        if self.target_return is None:
            returns = pf.as_returns().to_numpy().flatten()
            returns = returns[returns != 0]
            returns = returns[~np.isnan(returns)]
            return np.min(np.abs(returns))
        return self.target_return

    def _coerce_target_volatility(self, pf):
        """
        Coerces the target volatility parameter based on the given portfolio.

        Parameters
        ----------
        self : MVOptimizer
            The MVOptimizer instance.
        pf : Portfolio
            The portfolio for which the volatility needs to be coerced.

        Returns
        -------
        float
            The coerced target volatility value.
        """
        if self.target_risk is None:
            volatilities = np.std(pf.as_prices())
            return np.min(volatilities)
        return self.target_risk

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
        """Get the market_neutral parameter."""
        return self.market_neutral
    
    def _get_method(self):
        """Get current optimization method"""
        return this.method

    def _calculate_weights_by_risk(self, pf):
        """Calculate weights based on the risk of the portfolio.

        Parameters
        ----------
        self : MVOptimizer
            The MVOptimizer instance.
        pf : Portfolio
            The portfolio for which to calculate the weights.

        Returns
        -------
        tuple
            The calculated weights based on risk and optimizer metadata.
        """
        optimizer = self._get_optimizer(pf)
        target_volatility = self._coerce_target_volatility(pf)
        market_neutral = self.market_neutral

        weights_dict = optimizer.efficient_risk(
            target_volatility, market_neutral=market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": "efficient_risk",
            "target_volatility": target_volatility,
        }

        return weights, optimizer_metadata

    #FIXME Optimize es muy grande (Pasar a paquete)
    #FIXME Hacer que reciba todo lo que usa calculate_weights
    def _calculate_weights_by_return(self, pf):
        """Calculate weights based on the return of the portfolio.

        Parameters
        ----------
        self : MVOptimizer
            The MVOptimizer instance.
        pf : Portfolio
            The portfolio for which to calculate the weights.

        Returns
        -------
        tuple
            The calculated weights based on return and optimizer metadata.
        """
        optimizer = self._get_optimizer(pf)
        target_return = self._coerce_target_return(pf)
        market_neutral = self.market_neutral

        weights_dict = optimizer.efficient_return(
            target_return, market_neutral=market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": "efficient_return",
            "target_return": target_return,
        }

        return weights, optimizer_metadata

    def _calculate_weights_general(self, pf):
        """
        Calculate the optimal weights for a portfolio using a general
        optimization method.

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
        market_neutral = self._get_market_neutral()
        # market_neutral = self.market_neutral FIXME comentario pasarlo a acc

        weights_dict = getattr(optimizer, self._get_method())()  # TODO Pasarlo a acc
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": self.method,
        }

        return weights, optimizer_metadata

    def _calculate_weights(self, pf):
        """
        Calculate the optimal weights for a portfolio based on the
        specified method.

        Parameters
        ----------
        self : The MVOptimizer instance.
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        The optimal weights for the portfolio calculated using the specified
        method.
        """
        methods = {
            "efficient_risk": self._calculate_weights_by_risk,
            "efficient_return": self._calculate_weights_by_return,
            "general": self._calculate_weights_general,
        }
        method = methods[self.method]
        optimizer = self._get_optimizer(pf)
        target_return = self._get_target_return(pf)
        target_volatility = self._get_target_volatility(pf)
        weights = method(pf, optimizer=optimizer, target_return=target_return, target_volatility=target_volatility)

