# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import pypfopt

import numpy as np

from .core import Portfolio
from .utils.mabc import ModelABC, abstractmethod, hparam


# =============================================================================
# ABSTRACT OPTIMIZER
# =============================================================================


class OptimizerABC(ModelABC):
    @abstractmethod
    def optimize(self, pf):
        raise NotImplementedError()


# =============================================================================
# OPTIMIZER
# =============================================================================


class Markowitz(OptimizerABC):
    """Clasic Markowitz model.

    This method implements the  Clasic Model Markowitz 1952 in Mansini, R.,
    WLodzimierz, O., and Speranza, M. G. (2015). Linear and mixed
    integer programming for portfolio optimization. Springer and EURO: The
    Association of European Operational Research Societies

    """

    target_return = hparam(default=None)

    weight_bounds = hparam(default=(0, 1))
    market_neutral = hparam(default=False)

    returns = hparam(default="mah")
    returns_kw = hparam(factory=dict)

    covariance = hparam(default="sample_cov")
    covariance_kw = hparam(factory=dict)

    def _to_kwargs(self, pf):
        mu = pf.ereturns(self.returns, **self.returns_kw)
        cov = pf.covariance(self.covariance, **self.covariance_kw)
        return {
            "expected_returns": mu,
            "cov_matrix": cov,
            "weight_bounds": self.weight_bounds,
        }

    def _coerce_target_return(self, port):
        if self.target_return is None:
            returns = port.as_prices().to_numpy()
            return np.min(returns)
        return self.target_return

    def optimize(self, pf):
        kwargs = self._to_kwargs(pf)
        target_return = self._coerce_target_return(pf)

        ef = pypfopt.EfficientFrontier(**kwargs)

        weights = ef.efficient_return(target_return, self.market_neutral)

        weights_list = [weights[stock] for stock in self._prices_df.columns]

        metadata = dict(pf.metadata)
        metadata.update(
            optimizer=type(self).__name__,
            optimizer_kwargs={"target_return": target_return},
        )

        return Portfolio.from_dfkws(
            self._prices_df.copy(), weights_list, **metadata
        )


class BlackLitterman(OptimizerABC):
    risk_aversion = hparam(default=None)

    prior = hparam(default="equal")
    absolute_views = hparam(default=None)
    P = hparam(default=None)
    Q = hparam(default=None)
    covariance = hparam(default="sample_cov")
    covariance_kw = hparam(factory=dict)

    def _to_kwargs(self, pf):
        cov = pf.covariance(self.covariance, **self.covariance_kw)
        return {
            "cov_matrix": cov,
            "pi": self.prior,
            "absolute_views": self.absolute_views,
            "P": self.P,
            "Q": self.Q,
        }

    def optimize(self, pf):
        kwargs = self._to_kwargs(pf)
        risk_aversion = self.risk_aversion

        blm = pypfopt.BlackLittermanModel(**kwargs)
        weights = blm.bl_weights(risk_aversion)

        weights_list = [weights[stock] for stock in self._prices_df.columns]

        metadata = dict(pf.metadata)
        metadata.update(
            optimizer=type(self).__name__,
            optimizer_kwargs={"risk_aversion": risk_aversion},
        )

        return Portfolio.from_dfkws(
            self._prices_df.copy(), weights_list, **metadata
        )
