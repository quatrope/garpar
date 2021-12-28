# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.black_litterman import BlackLittermanModel

from .core import Portfolio
from .utils.mabc import ModelABC, abstractmethod, hparam

# =============================================================================
# PYPORTFOLIO WRAPPER FUNCTIONS
# =============================================================================


def mean_historical_return(portfolio):
    return expected_returns.mean_historical_return(portfolio._df)


def sample_covariance(portfolio):
    return risk_models.sample_cov(portfolio._df)


# =============================================================================
# ABSTRACT OPTIMIZER
# =============================================================================


class OptimizerABC(ModelABC):
    @abstractmethod
    def serialize(self, port):
        raise NotImplementedError()

    @abstractmethod
    def deserialize(self, port, weights):
        raise NotImplementedError()

    @abstractmethod
    def optimize(self, port):
        raise NotImplementedError()


class Markowitz(OptimizerABC):

    weight_bounds = hparam(default=(0, 1))
    market_neutral = hparam(default=False)

    def serialize(self, port):
        mu = mean_historical_return(port)
        cov = sample_covariance(port)
        return {
            "expected_returns": mu,
            "cov_matrix": cov,
            "weight_bounds": self.weight_bounds,
        }

    def deserialize(self, port, weights):
        weights_list = [weights[stock] for stock in port._df.columns]
        return Portfolio(port._df.copy(), weights_list)

    def optimize(self, port, target_return):
        kwargs = self.serialize(port)

        ef = EfficientFrontier(**kwargs)
        weights = ef.efficient_return(target_return, self.market_neutral)

        return self.deserialize(port, weights)


class BlackLitterman(OptimizerABC):

    prior = hparam(default="equal")
    absolute_views = hparam(default=None)
    P = hparam(default=None)
    Q = hparam(default=None)

    def serialize(self, port):
        cov = sample_covariance(port)
        return {
            "cov_matrix": cov,
            "pi": self.prior,
            "absolute_views": self.absolute_views,
            "P": self.P,
            "Q": self.Q,
        }

    def deserialize(self, port, weights):
        weights_list = [weights[stock] for stock in port._df.columns]
        return Portfolio(port._df.copy(), weights_list)

    def optimize(self, port, risk_aversion=None):
        kwargs = self.serialize(port)

        blm = BlackLittermanModel(**kwargs)
        weights = blm.bl_weights(risk_aversion)

        return self.deserialize(port, weights)
