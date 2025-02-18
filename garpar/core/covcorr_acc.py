# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Covariance Accessor.

The covariance accessor module provides an accessor class to compute
various covariance matrices. The module also defines the CorrelationAccessor
class, which provides methods for calculating different correlation matrices.

Key Features:
    - Correlation and covariance analysis

Example
-------
    >>> import garpar
    >>> ss = garpar.mkss(prices=[...])
    >>> ss.covariance.sample_cov()
    >>> ss.covariance.exp_cov()
    >>> ss.covariance.semi_cov()
    >>> ss.covariance.ledoit_wolf_cov()
    >>> ss.covariance.oracle_approximating_cov()
    >>> ss.correlation.sample_corr()
    >>> ss.correlation.exp_corr()
    >>> ss.correlation.semi_corr()
    >>> ss.correlation.ledoit_wolf_corr()
    >>> ss.correlation.oracle_approximating_corr()

"""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

from pypfopt import risk_models

from ..utils import AccessorABC

# =============================================================================
# COVARIANCE ACCESSOR
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class CovarianceAccessor(AccessorABC):
    """Accessor class for calculating various covariance matrices."""

    _default_kind = "sample_cov"

    _ss = attr.ib()

    def sample_cov(self, **kwargs):
        """Compute the sample covariance matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to `risk_models.sample_cov`.

        Returns
        -------
        pandas.DataFrame
            Sample covariance matrix.
        """
        return risk_models.sample_cov(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )

    def exp_cov(self, **kwargs):
        """Compute the exponentially-weighted covariance matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to `risk_models.exp_cov`.

        Returns
        -------
        pandas.DataFrame
            Exponentially-weighted covariance matrix.
        """
        return risk_models.exp_cov(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )

    def semi_cov(self, **kwargs):
        """Compute the semi-covariance matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            `risk_models.semicovariance`.

        Returns
        -------
        pandas.DataFrame
            Semi-covariance matrix.
        """
        return risk_models.semicovariance(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )

    def ledoit_wolf_cov(self, shrinkage_target="constant_variance", **kwargs):
        """Compute the Ledoit-Wolf covariance matrix.

        Compute the Ledoit-Wolf covariance matrix
        with optional shrinkage target

        Parameters
        ----------
        shrinkage_target : str, optional
            Shrinkage target for Ledoit-Wolf covariance estimation, default is
            "constant_variance".
        **kwargs
            Additional keyword arguments passed to
            `risk_models.CovarianceShrinkage.ledoit_wolf`.

        Returns
        -------
        pandas.DataFrame
            Ledoit-Wolf covariance matrix.
        """
        covshrink = risk_models.CovarianceShrinkage(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )
        return covshrink.ledoit_wolf(shrinkage_target=shrinkage_target)

    def oracle_approximating_cov(self, **kwargs):
        """Compute the Oracle-approximating covariance matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            `risk_models.CovarianceShrinkage.oracle_approximating`.

        Returns
        -------
        pandas.DataFrame
            Oracle-approximating covariance matrix.
        """
        covshrink = risk_models.CovarianceShrinkage(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )
        return covshrink.oracle_approximating()


# =============================================================================
# CORRELATION ACCESSOR
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class CorrelationAccessor(AccessorABC):
    """Accessor class for calculating various correlation matrices."""

    _default_kind = "sample_corr"

    _ss = attr.ib()

    def sample_corr(self, **kwargs):
        """Compute the sample correlation matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            `self._ss.covariance.sample_cov`.

        Returns
        -------
        pandas.DataFrame
            Sample correlation matrix.
        """
        cov = self._ss.covariance.sample_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def exp_corr(self, **kwargs):
        """Compute the exponentially-weighted correlation matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            `self._ss.covariance.exp_cov`.

        Returns
        -------
        pandas.DataFrame
            Exponentially-weighted correlation matrix.
        """
        cov = self._ss.covariance.exp_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def semi_corr(self, **kwargs):
        """Compute the semi-correlation matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            `self._ss.covariance.semi_cov`.

        Returns
        -------
        pandas.DataFrame
            Semi-correlation matrix.
        """
        cov = self._ss.covariance.semi_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def ledoit_wolf_corr(self, **kwargs):
        """Compute the Ledoit-Wolf correlation matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            `self._ss.covariance.ledoit_wolf_cov`.

        Returns
        -------
        pandas.DataFrame
            Ledoit-Wolf correlation matrix.
        """
        cov = self._ss.covariance.ledoit_wolf_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def oracle_approximating_corr(self, **kwargs):
        """Compute the Oracle-approximating correlation matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            `self._ss.covariance.oracle_approximating_cov`.

        Returns
        -------
        pandas.DataFrame
            Oracle-approximating correlation matrix.
        """
        cov = self._ss.covariance.oracle_approximating_cov(**kwargs)
        return risk_models.cov_to_corr(cov)
