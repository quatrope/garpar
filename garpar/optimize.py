# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Portfolio optimizers."""

import pypfopt

import numpy as np

from .core import Portfolio
from .utils import Bunch, mabc


# =============================================================================
# ABSTRACT OPTIMIZER
# =============================================================================


class OptimizerABC(mabc.ModelABC):
    """Abstract optimizer."""

    @mabc.abstractmethod
    def _calculate_weights(self, pf):
        raise NotImplementedError()

    def optimize(self, pf):
        """Skeleton for optimize."""
        weights, metadata = self._calculate_weights(pf)
        return pf.copy(weights=weights, optimizer=metadata)

# =============================================================================
# OPTIMIZER
# =============================================================================

class MVOptimizer(OptimizerABC):
    """Mean Variance Optimizer."""

    weight_bounds = mabc.hparam(default=(0, 1))

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
        if self.target_return is None:
            returns = pf.as_returns().to_numpy().flatten()
            returns = returns[returns != 0]
            returns = returns[~np.isnan(returns)]
            return np.min(np.abs(returns))
        return self.target_return

    def _coerce_target_volatility(self, pf):
        if self.target_risk is None:
            volatilities = np.std(pf.as_prices()) # Se revisó y se usa efectivamente la deviacion estandar. Ya que despues usa el cuadrado de este valor. Se entiende este parametro como volatilidad?
            return np.min(volatilities)
        return self.target_risk

    def _get_optimizer(self, pf):
        expected_returns = pf.ereturns(self.returns, **self.returns_kw)
        cov_matrix = pf.covariance(self.covariance, **self.covariance_kw)
        weight_bounds = self.weight_bounds
        optimizer = pypfopt.EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            weight_bounds=weight_bounds,
        )
        return optimizer

    def __calculate_weights_by_risk(self, pf):
        optimizer = self._get_optimizer(pf)
        target_volatility = self._coerce_target_volatility(pf)
        market_neutral = self.market_neutral

        weights_dict = optimizer.efficient_risk(target_volatility, market_neutral=market_neutral)
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": "efficient_risk",
            "target_volatility": target_volatility,
        }

        return weights, optimizer_metadata

    def __calculate_weights_by_return(self, pf):
        optimizer = self._get_optimizer(pf)
        target_return = self._coerce_target_return(pf)
        market_neutral = self.market_neutral

        weights_dict = optimizer.efficient_return(target_return, market_neutral=market_neutral)
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": "efficient_return",
            "target_return": target_return,
        }

        return weights, optimizer_metadata

    def __calculate_weights_general(self, pf):
        optimizer = self._get_optimizer(pf)
        market_neutral = self.market_neutral

        weights_dict = getattr(optimizer, self.method)()
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": self.method,
        }

        return weights, optimizer_metadata

    def _calculate_weights(self, pf):
        if self.method == "efficient_risk":
            return self.__calculate_weights_by_risk(pf)
        elif self.method == "efficient_return":
            return self.__calculate_weights_by_return(pf)
        else:
            return self.__calculate_weights_general(pf)

class Markowitz(OptimizerABC):
    """Clasic Markowitz model.

    This method implements the  Clasic Model Markowitz 1952 in Mansini, R.,
    WLodzimierz, O., and Speranza, M. G. (2015). Linear and mixed
    integer programming for portfolio optimization. Springer and EURO: The
    Association of European Operational Research Societies

    """

    target_return = mabc.hparam(default=None)

    weight_bounds = mabc.hparam(default=(0, 1))
    market_neutral = mabc.hparam(default=False)

    returns = mabc.hparam(default="mah")
    returns_kw = mabc.hparam(factory=dict)

    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)

    def _get_optimizer(self, pf):
        expected_returns = pf.ereturns(self.returns, **self.returns_kw)
        cov_matrix = pf.covariance(self.covariance, **self.covariance_kw)
        weight_bounds = self.weight_bounds
        optimizer = pypfopt.EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            weight_bounds=weight_bounds,
        )
        return optimizer

    def _coerce_target_return(self, pf):
        if self.target_return is None:
            returns = pf.as_returns().to_numpy().flatten()
            returns = returns[returns != 0]
            returns = returns[~np.isnan(returns)]
            return np.min(np.abs(returns))
        return self.target_return

    def _calculate_weights(self, pf):
        optimizer = self._get_optimizer(pf)
        target_return = self._coerce_target_return(pf)
        market_neutral = self.market_neutral

        weights_dict = optimizer.efficient_return(
            target_return, market_neutral=market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            "target_return": target_return,
        }

        return weights, optimizer_metadata

class BlackLitterman(OptimizerABC):
    """Classic Black Litterman model."""

    risk_aversion = mabc.hparam(default=None)
    prior = mabc.hparam(default="equal")
    absolute_views = mabc.hparam(default=None)
    P = mabc.hparam(default=None)
    Q = mabc.hparam(default=None)
    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)

    def _get_optimizer(self, pf):
        cov = pf.covariance(self.covariance, **self.covariance_kw)
        prior = self.prior
        absolute_views = self.absolute_views
        P = self.P
        Q = self.Q

        return pypfopt.BlackLittermanModel(
            cov_matrix=cov, pi=prior, absolute_views=absolute_views, P=P, Q=Q
        )

    def _calculate_weights(self, pf):
        blm = self._get_optimizer(pf)
        risk_aversion = self.risk_aversion

        weights_dict = blm.bl_weights(risk_aversion)
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            "risk_aversion": risk_aversion,
        }

        return weights, optimizer_metadata
