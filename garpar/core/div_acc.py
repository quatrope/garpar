# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Diversification Accessor.

The diversification accessor module provides an accessor class to compute
diversification metrics for a given stocks set.

Key Features:
    - Diversification metrics calculation

Example
-------
    >>> import garpar
    >>> ss = garpar.mkss(prices=[...])
    >>> ss.diversification.ratio()
    >>> ss.diversification.mrc()
    >>> ss.diversification.pdi()

"""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np

from sklearn.decomposition import PCA

from . import coercer_mixin
from ..utils import AccessorABC

# =============================================================================
# DIVERSIFICATION
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class DiversificationMetricsAccessor(AccessorABC, coercer_mixin.CoercerMixin):
    """A class to calculate various diversification metrics for a StocksSet."""

    _default_kind = "ratio"

    _ss = attr.ib()

    def ratio(self, *, covariance="sample_cov", covariance_kw=None):
        """Calculate the diversification ratio.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix,
            by default "sample_cov".
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method.

        Returns
        -------
        float
            The diversification ratio.

        Example
        -------
        >>> accessor = DiversificationMetricsAccessor(ss)
        >>> ratio = accessor.ratio()
        """
        weights = self._ss.scale_weights().weights
        ret_std = self._ss.as_returns().std()
        ss_variance = self._ss.risk.ss_var(
            covariance=covariance, covariance_kw=covariance_kw
        )

        return np.sum(weights * ret_std) / np.sqrt(ss_variance)

    def mrc(self, *, covariance="sample_cov", covariance_kw=None):
        """Calculate the marginal risk contribution.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix,
            by default "sample_cov".
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method.

        Returns
        -------
        Series
            The marginal risk contribution.

        Example
        -------
        >>> accessor = DiversificationMetricsAccessor(ss)
        >>> mrc = accessor.mrc()
        """
        weights = self._ss.scale_weights().weights

        cov_matrix = self.coerce_covariance_matrix(
            covariance, covariance_kw, asarray=False
        )

        ss_variance = self._ss.risk.ss_var(
            covariance=cov_matrix, covariance_kw=None
        )

        result = np.sum(cov_matrix * weights, axis=1) / np.sqrt(ss_variance)
        result.name = "MRC"
        return result

    def pdi(
        self,
        *,
        n_components=None,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        """Calculate the stocks set diversification index.

        Parameters
        ----------
        n_components : int, optional
            Number of components to keep, by default None.
        whiten : bool, optional
            Whether to whiten the components, by default False.
        svd_solver : str, optional
            The solver to use for SVD, by default "auto".
        tol : float, optional
            Tolerance for singular values, by default 0.0.
        iterated_power : int or str, optional
            Number of iterations for power method, by default "auto".
        n_oversamples : int, optional
            Number of oversamples for randomized SVD, by default 10.
        power_iteration_normalizer : str, optional
            Normalizer for power iterations, by default "auto".
        random_state : int, RandomState instance or None, optional
            Seed or random number generator for reproducibility,
            by default None.

        Returns
        -------
        float
            The stocks set diversification index.

        Example
        -------
        >>> accessor = DiversificationMetricsAccessor(ss)
        >>> pdi = accessor.pdi()
        """
        returns = self._ss.as_returns()

        pca = PCA(
            n_components=n_components,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        ).fit(returns.T)

        rsi = pca.explained_variance_ratio_
        n_components = np.arange(1, pca.n_components_ + 1)

        pdi = 2 * np.sum(n_components * rsi) - 1
        return pdi

    def zheng_entropy(self):
        """Calculate Zheng's entropy.

        Returns
        -------
        float
            Zheng's entropy.

        Example
        -------
        >>> accessor = DiversificationMetricsAccessor(ss)
        >>> entropy = accessor.zheng_entropy()
        """
        weights = self._ss.scale_weights().weights
        return -np.sum(weights * np.log(weights))

    def cross_entropy(self, benchmark_weights=None):
        """Calculate cross entropy.

        Parameters
        ----------
        benchmark_weights : array-like, optional
            The benchmark weights to compare against.

        Returns
        -------
        float
            Cross entropy.

        Example
        -------
        >>> accessor = DiversificationMetricsAccessor(ss)
        >>> cross_entropy = accessor.cross_entropy()
        """
        weights = self._ss.scale_weights().weights
        benchmark_weights = self.coerce_weights(benchmark_weights)
        return np.sum(benchmark_weights * np.log(benchmark_weights / weights))

    def ke_zang_entropy(self, *, covariance="sample_cov", covariance_kw=None):
        """Calculate Ke and Zang's entropy.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix,
            by default "sample_cov".
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method.

        Returns
        -------
        float
            Ke and Zang's entropy.

        Example
        -------
        >>> accessor = DiversificationMetricsAccessor(ss)
        >>> entropy = accessor.ke_zang_entropy()
        """
        ss_var = self._ss.risk.ss_var(
            covariance=covariance, covariance_kw=covariance_kw
        )
        entropy = self.zheng_entropy()

        return ss_var + entropy
