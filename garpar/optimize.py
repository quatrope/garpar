# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Portfolio optimizers."""

import attr

import numpy as np

import pypfopt

from .utils import mabc

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
        """Check if the 'family' attribute is of type string."""
        if cls.family is _Unknow or not isinstance(cls.family, str):
            cls_name = cls.__name__
            raise TypeError(f"'{cls_name}.family' must be redefined as string")

    @mabc.abstractmethod
    def _calculate_weights(self, pf):
        """Boilerplate method to calculate portfolio weights."""
        raise NotImplementedError()

    def optimize(self, pf):
        """Optimize the given portfolio.

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
        """Get the family of the optimizer.

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

    def __calculate_weights_by_risk(self, pf):
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

    def __calculate_weights_by_return(self, pf):
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

    def __calculate_weights_general(self, pf):
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
        if self.method == "efficient_risk":
            return self.__calculate_weights_by_risk(pf)
        elif self.method == "efficient_return":
            return self.__calculate_weights_by_return(pf)
        else:
            return self.__calculate_weights_general(pf)


class Markowitz(MVOptimizer):
    """Classic Markowitz model.

    This method implements the Classic Model Markowitz 1952 in Mansini, R.,
    WLodzimierz, O., and Speranza, M. G. (2015). Linear and mixed
    integer programming for portfolio optimization. Springer and EURO: The
    Association of European Operational Research Societies
    """

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
        # market_neutral = self.market_neutral

        weights_dict = optimizer.efficient_return(
            target_return, market_neutral=market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            "target_return": target_return,
        }

        return weights, optimizer_metadata


@attr.define(repr=False)
class BlackLitterman(OptimizerABC):
    """Classic Black Litterman model.

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
        """Get the pypfopt BlackLittermanModel optimizer.

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
        """Calculate the optimal weights for the portfolio.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        tuple
            Tuple containing the optimal weights and optimizer metadata.
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
