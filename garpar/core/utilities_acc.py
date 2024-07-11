# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Utilies Accessor."""

import attr

from pypfopt import objective_functions

from . import _mixins
from ..utils import accabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class UtilitiesAccessor(accabc.AccessorABC, _mixins.CoercerMixin):
    """Accessor for various utility and performance metrics.

    The UtilitiesAccessor class provides methods to compute ex-ante tracking error,
    ex-post tracking error, portfolio return, and quadratic utility for a given
    portfolio.

    Attributes
    ----------
    _default_kind : str
        The default kind of performance measure, default is "pf_return".
    _pf : attr.ib
        The portfolio object containing weights and other attributes.

    Methods
    -------
    ex_ante_tracking_error(covariance='sample_cov', covariance_kw=None, benchmark_weights=None):
        Computes the ex-ante tracking error of the portfolio.
    ex_post_tracking_error(historic_returns='capm', benchmark_returns='capm', historic_returns_kw=None, benchmark_returns_kw=None):
        Computes the ex-post tracking error of the portfolio.
    portfolio_return(expected_returns='capm', expected_returns_kw=None, negative=True):
        Computes the expected return of the portfolio.
    quadratic_utility(expected_returns='capm', covariance='sample_cov', risk_aversion=0.5, expected_returns_kw=None, covariance_kw=None, **kwargs):
        Computes the quadratic utility of the portfolio.
    """

    _default_kind = "pf_return"

    _pf = attr.ib()

    def ex_ante_tracking_error(
        self,
        *,
        covariance="sample_cov",
        covariance_kw=None,
        benchmark_weights=None,
    ):
        """Compute the ex-ante tracking error of the portfolio.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix, by default 'sample_cov'.
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method, by default None.
        benchmark_weights : array-like, optional
            The weights of the benchmark portfolio, by default None.

        Returns
        -------
        float
            The ex-ante tracking error of the portfolio.

        Examples
        --------
        >>> accessor = UtilitiesAccessor(pf)
        >>> error = accessor.ex_ante_tracking_error()
        """
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)
        benchmark_weights = self.coerce_weights(benchmark_weights)

        return objective_functions.ex_ante_tracking_error(
            self._pf._weights,
            cov_matrix=cov_matrix,
            benchmark_weights=benchmark_weights,
        )

    def ex_post_tracking_error(
        self,
        *,
        historic_returns="capm",
        benchmark_returns="capm",
        historic_returns_kw=None,
        benchmark_returns_kw=None,
    ):
        """Compute the ex-post tracking error of the portfolio.

        Parameters
        ----------
        historic_returns : str, optional
            The method to compute the historic returns, by default 'capm'.
        benchmark_returns : str, optional
            The method to compute the benchmark returns, by default 'capm'.
        historic_returns_kw : dict, optional
            Additional keyword arguments for the historic returns method, by default None.
        benchmark_returns_kw : dict, optional
            Additional keyword arguments for the benchmark returns method, by default None.

        Returns
        -------
        float
            The ex-post tracking error of the portfolio.

        Examples
        --------
        >>> accessor = UtilitiesAccessor(pf)
        >>> error = accessor.ex_post_tracking_error()
        """
        historic_returns = self.coerce_expected_returns(
            historic_returns, historic_returns_kw
        )
        benchmark_returns = self.coerce_expected_returns(
            benchmark_returns, benchmark_returns_kw
        )
        return objective_functions.ex_post_tracking_error(
            self._pf._weights,
            historic_returns=historic_returns,
            benchmark_returns=benchmark_returns,
        )

    def portfolio_return(
        self,
        *,
        expected_returns="capm",
        expected_returns_kw=None,
        negative=True,
    ):
        """Compute the expected return of the portfolio.

        Parameters
        ----------
        expected_returns : str, optional
            The method to compute the expected returns, by default 'capm'.
        expected_returns_kw : dict, optional
            Additional keyword arguments for the expected returns method, by default None.
        negative : bool, optional
            Whether to return the negative of the expected return, by default True.

        Returns
        -------
        float
            The expected return of the portfolio.

        Examples
        --------
        >>> accessor = UtilitiesAccessor(pf)
        >>> ret = accessor.portfolio_return()
        """
        expected_returns = self.coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        return objective_functions.portfolio_return(
            self._pf._weights,
            expected_returns=expected_returns,
            negative=negative,
        )

    pf_return = portfolio_return

    def quadratic_utility(
        self,
        *,
        expected_returns="capm",
        covariance="sample_cov",
        risk_aversion=0.5,
        expected_returns_kw=None,
        covariance_kw=None,
        **kwargs,
    ):
        """Compute the quadratic utility of the portfolio.

        Parameters
        ----------
        expected_returns : str, optional
            The method to compute the expected returns, by default 'capm'.
        covariance : str, optional
            The method to compute the covariance matrix, by default 'sample_cov'.
        risk_aversion : float, optional
            The risk aversion coefficient, by default 0.5.
        expected_returns_kw : dict, optional
            Additional keyword arguments for the expected returns method, by default None.
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method, by default None.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            The quadratic utility of the portfolio.

        Examples
        --------
        >>> accessor = UtilitiesAccessor(pf)
        >>> utility = accessor.quadratic_utility()
        """
        expected_returns = self.coerce_expected_returns(
            expected_returns, expected_returns_kw
        )
        cov_matrix = self.coerce_covariance_matrix(covariance, covariance_kw)

        return objective_functions.quadratic_utility(
            self._pf._weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_aversion=risk_aversion,
            **kwargs,
        )

    qutility = quadratic_utility
