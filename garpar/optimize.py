# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import abc

import attr

from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier


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


class OptimizerABC(metaclass=abc.ABCMeta):

    # parametros  = hparam(...)

    @abc.abstractmethod
    def serialize(self, port):
        ...

    @abc.abstractmethod
    def deserialize(self):
        ...

    @abc.abstractmethod
    def optimize():
        ...


@attr.s(frozen=True)
class Markowitz(OptimizerABC):

    weight_bounds = attr.ib(default=(0, 1))
    market_neutral = attr.ib(default=False)

    def serialize(self, port):
        mu = mean_historical_return(port)
        cov = sample_covariance(port)
        return {
            "expected_returns": mu,
            "cov_matrix": cov,
            "weight_bounds": self.weight_bounds,
        }

    def deserialize(self, port, weights):
        pass

    def optimize(self, port, target_return):
        kwargs = self.serialize(port)

        ef = EfficientFrontier(**kwargs)
        weights = ef.efficient_return(target_return, self.market_neutral)

        # new_port = self.deserialize(port, weights)
        return weights
