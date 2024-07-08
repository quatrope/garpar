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

_Unknow = object()


class OptimizerABC(mabc.ModelABC):
    """
    Abstract base class for portfolio optimizers.

    Attributes
    ----------
    family : str
        The family of the optimizer.

    Methods
    -------
    optimize(pf)
        Optimize the given portfolio.
    get_optimizer_family()
        Get the family of the optimizer.
    """

    family = _Unknow

    def __init_subclass__(cls):
        if cls.family is _Unknow or not isinstance(cls.family, str):
            cls_name = cls.__name__
            raise TypeError(f"'{cls_name}.family' must be redefined as string")

    @mabc.abstractmethod
    def _calculate_weights(self, pf):
        raise NotImplementedError()

    def optimize(self, pf):
        """
        Optimize the given portfolio.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        Portfolio
            A new portfolio with optimized weights.
        """
        weights, metadata = self._calculate_weights(pf)
        return pf.copy(weights=weights, optimizer=metadata)

    @classmethod
    def get_optimizer_family(cls):
        """
        Get the family of the optimizer.

        Returns
        -------
        str
            The family of the optimizer.
        """
        return cls.family


# =============================================================================
# OPTIMIZER
# =============================================================================

class MeanVarianceFamilyMixin:
    """Mixin class for mean-variance family optimizers."""

    family = "mean-variance"


class Markowitz(MeanVarianceFamilyMixin, OptimizerABC):
    """Classic Markowitz model.

    This method implements the Classic Model Markowitz 1952 in Mansini, R.,
    WLodzimierz, O., and Speranza, M. G. (2015). Linear and mixed
    integer programming for portfolio optimization. Springer and EURO: The
    Association of European Operational Research Societies

    Attributes
    ----------
    target_return : float, optional
        The target return for the portfolio.
    weight_bounds : tuple of float, optional
        The bounds for asset weights (default is (0, 1)).
    market_neutral : bool, optional
        Whether to enforce a market neutral portfolio (default is False).
    returns : str, optional
        The method to calculate expected returns (default is "mah").
    returns_kw : dict, optional
        Additional keyword arguments for returns calculation.
    covariance : str, optional
        The method to calculate covariance matrix (default is "sample_cov").
    covariance_kw : dict, optional
        Additional keyword arguments for covariance calculation.
    optimize_options : list of str
        Available optimization strategies.
    """

    target_return = mabc.hparam(default=None)
    weight_bounds = mabc.hparam(default=(0, 1))
    market_neutral = mabc.hparam(default=False)
    returns = mabc.hparam(default="mah")
    returns_kw = mabc.hparam(factory=dict)
    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)
    optimize_options = [
        "min_volatility",
        "max_sharpe",
        "max_quadratic_utility",
        "efficient_risk",
        "efficient_return",
    ]

    def _get_optimizer(self, pf):
        """
        Get the pypfopt EfficientFrontier optimizer.

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

    def _coerce_target_return(self, pf):
        """
        Coerce the target return.

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
            returns = pf.as_returns().to_numpy()
            return np.min(np.abs(returns))
        return self.target_return

    def _calculate_weights(self, pf):
        """
        Calculate the optimal weights for the portfolio.

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

    def _optimize(self, pf, *, op="max_sharpe", **kwargs):
        """
        Optimize the portfolio using the specified strategy.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.
        op : str, optional
            The optimization strategy to use (default is "max_sharpe").
        **kwargs
            Additional keyword arguments for the optimization.

        Returns
        -------
        tuple
            A tuple containing the optimal weights and optimizer metadata.
        """
        optimizer = self._get_optimizer(pf)

        global_min_volatility = np.sqrt(
            1 / np.sum(np.linalg.pinv(optimizer.cov_matrix))
        )

        kwargs.setdefault(
            "target_return", optimizer.deepcopy()._max_return() - 0.0001
        )
        kwargs.setdefault("target_risk", global_min_volatility)

        if op not in self.optimize_options:
            print("Not a valid option, using max_sharpe instead")
            op = "max_sharpe"

        if op == "efficient_risk":
            optimizer = self._get_optimizer(pf)
            weights = optimizer.efficient_risk(kwargs["target_risk"])
            optimizer_metadata = {
                "name": type(self).__name__,
                "target_risk": kwargs["target_risk"],
            }

            return weights, optimizer_metadata

        if op == "efficient_return":
            optimizer = self._get_optimizer(pf)
            weights = optimizer.efficient_return(kwargs["target_return"])
            optimizer_metadata = {
                "name": type(self).__name__,
                "target_return": kwargs["target_return"],
            }

            return weights, optimizer_metadata

        weights = getattr(optimizer, op)()

        optimizer_metadata = {
            "name": type(self).__name__,
        }

        return weights, optimizer_metadata


class BlackLitterman(OptimizerABC):
    """
    Classic Black Litterman model.

    Attributes
    ----------
    family : str
        The family of the optimizer (set to "black-litterman").
    risk_aversion : float, optional
        The risk aversion parameter.
    prior : str or array-like, optional
        The prior for expected returns (default is "equal").
    absolute_views : dict, optional
        Absolute views on assets.
    P : array-like, optional
        Pick matrix for relative views.
    Q : array-like, optional
        View matrix for relative views.
    covariance : str, optional
        The method to calculate covariance matrix (default is "sample_cov").
    covariance_kw : dict, optional
        Additional keyword arguments for covariance calculation.
    """

    family = "black-litterman"

    risk_aversion = mabc.hparam(default=None)
    prior = mabc.hparam(default="equal")
    absolute_views = mabc.hparam(default=None)
    P = mabc.hparam(default=None)
    Q = mabc.hparam(default=None)
    covariance = mabc.hparam(default="sample_cov")
    covariance_kw = mabc.hparam(factory=dict)

    def _get_optimizer(self, pf):
        """
        Get the pypfopt BlackLittermanModel optimizer.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        pypfopt.BlackLittermanModel
            The configured optimizer.
        """
        cov = pf.covariance(self.covariance, **self.covariance_kw)
        prior = self.prior
        absolute_views = self.absolute_views
        P = self.P
        Q = self.Q

        return pypfopt.BlackLittermanModel(
            cov_matrix=cov, pi=prior, absolute_views=absolute_views, P=P, Q=Q
        )

    def _calculate_weights(self, pf):
        """
        Calculate the optimal weights for the portfolio.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        tuple
            A tuple containing the optimal weights and optimizer metadata.
        """
        blm = self._get_optimizer(pf)
        risk_aversion = self.risk_aversion

        weights_dict = blm.bl_weights(risk_aversion)
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            "risk_aversion": risk_aversion,
        }

        return weights, optimizer_metadata