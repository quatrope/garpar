# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import pypfopt

from .core import Portfolio
from .utils.mabc import ModelABC, abstractmethod, hparam


# =============================================================================
# ABSTRACT OPTIMIZER
# =============================================================================


class OptimizerABC(ModelABC):


    @abstractmethod
    def optimize(self, port):
        raise NotImplementedError()


# =============================================================================
# OPTIMIZER
# =============================================================================


class Markowitz(OptimizerABC):

    weight_bounds = hparam(default=(0, 1))
    market_neutral = hparam(default=False)

    returns = hparam(default="mah")
    returns_kw = hparam(factory=dict)

    covariance = hparam(default="sample_cov")
    covariance_kw = hparam(factory=dict)

    def serialize(self, port):
        mu = port.ereturns(self.returns, **self.returns_kw)
        cov = port.covariance(self.covariance, **self.covariance_kw)
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

        ef = pypfopt.EfficientFrontier(**kwargs)

        weights = ef.efficient_return(target_return, self.market_neutral)

        return self.deserialize(port, weights)


class BlackLitterman(OptimizerABC):

    prior = hparam(default="equal")
    absolute_views = hparam(default=None)
    P = hparam(default=None)
    Q = hparam(default=None)
    covariance = hparam(default="sample_cov")
    covariance_kw = hparam(factory=dict)

    def serialize(self, port):
        cov = port.covariance(self.covariance, **self.covariance_kw)
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

        blm = pypfopt.BlackLittermanModel(**kwargs)
        weights = blm.bl_weights(risk_aversion)

        return self.deserialize(port, weights)
