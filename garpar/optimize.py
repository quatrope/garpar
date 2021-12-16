# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

from pypfopt import expected_returns, risk_models, EfficientFrontier

from .utils.mabc import ModelABC, hparam, abstractmethod

from .portfolio import Portfolio

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
    def optimize(self, port, target_return):
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
        pass

    def optimize(self, port, target_return):
        kwargs = self.serialize(port)

        ef = EfficientFrontier(**kwargs)
        weights = ef.efficient_return(target_return, self.market_neutral)

        weights_list = [weights[stock] for stock in port._df.columns]
        return Portfolio(port._df.copy(), weights_list)
