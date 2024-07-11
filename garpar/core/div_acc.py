# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Diversification Accessor."""

import attr

import numpy as np

# import scipy.stats

from sklearn.decomposition import PCA

from . import _mixins
from ..utils import accabc

# =============================================================================
# DIVERSIFICATION
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class DiversificationAccessor(accabc.AccessorABC, _mixins.CoercerMixin):
    """A class to calculate various diversification metrics for a portfolio.

    Attributes
    ----------
    _default_kind : str
        Default kind of diversification metric.
    _pf : Portfolio
        The portfolio object.

    Methods
    -------
    ratio(covariance="sample_cov", covariance_kw=None)
        Calculate the diversification ratio.
    mrc(covariance="sample_cov", covariance_kw=None)
        Calculate the marginal risk contribution.
    pdi(n_components=None, whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto",
        n_oversamples=10, power_iteration_normalizer="auto", random_state=None)
        Calculate the portfolio diversification index.
    zheng_entropy()
        Calculate Zheng's entropy.
    cross_entropy(benchmark_weights=None)
        Calculate cross entropy.
    ke_zang_entropy(covariance="sample_cov", covariance_kw=None)
        Calculate Ke and Zang's entropy.
    """

    _default_kind = "ratio"

    _pf = attr.ib()

    def ratio(self, *, covariance="sample_cov", covariance_kw=None):
        """Calculate the diversification ratio.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix, by default "sample_cov".
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method.

        Returns
        -------
        float
            The diversification ratio.

        Examples
        --------
        >>> accessor = DiversificationAccessor(pf)
        >>> ratio = accessor.ratio()
        """
        weights = self._pf.scale_weights().weights
        ret_std = self._pf.as_returns().std()
        pf_variance = self._pf.risk.pf_var(
            covariance=covariance, covariance_kw=covariance_kw
        )

        return np.sum(weights * ret_std) / np.sqrt(pf_variance)

    def mrc(self, *, covariance="sample_cov", covariance_kw=None):
        """Calculate the marginal risk contribution.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix, by default "sample_cov".
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method.

        Returns
        -------
        Series
            The marginal risk contribution.

        Examples
        --------
        >>> accessor = DiversificationAccessor(pf)
        >>> mrc = accessor.mrc()
        """
        weights = self._pf.scale_weights().weights

        cov_matrix = self.coerce_covariance_matrix(
            covariance, covariance_kw, asarray=False
        )

        pf_variance = self._pf.risk.pf_var(
            covariance=cov_matrix, covariance_kw=None
        )

        result = np.sum(cov_matrix * weights, axis=1) / np.sqrt(pf_variance)
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
        """Calculate the portfolio diversification index.

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
            Seed or random number generator for reproducibility, by default None.

        Returns
        -------
        float
            The portfolio diversification index.

        Examples
        --------
        >>> accessor = DiversificationAccessor(pf)
        >>> pdi = accessor.pdi()
        """
        returns = self._pf.as_returns()

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

        Examples
        --------
        >>> accessor = DiversificationAccessor(pf)
        >>> entropy = accessor.zheng_entropy()
        """
        weights = self._pf.scale_weights().weights
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

        Examples
        --------
        >>> accessor = DiversificationAccessor(pf)
        >>> cross_entropy = accessor.cross_entropy()
        """
        weights = self._pf.scale_weights().weights
        benchmark_weights = self.coerce_weights(benchmark_weights)
        return np.sum(benchmark_weights * np.log(benchmark_weights / weights))

    def ke_zang_entropy(self, *, covariance="sample_cov", covariance_kw=None):
        """Calculate Ke and Zang's entropy.

        Parameters
        ----------
        covariance : str, optional
            The method to compute the covariance matrix, by default "sample_cov".
        covariance_kw : dict, optional
            Additional keyword arguments for the covariance method.

        Returns
        -------
        float
            Ke and Zang's entropy.

        Examples
        --------
        >>> accessor = DiversificationAccessor(pf)
        >>> entropy = accessor.ke_zang_entropy()
        """
        pf_var = self._pf.risk.pf_var(
            covariance=covariance, covariance_kw=covariance_kw
        )
        entropy = self.zheng_entropy()

        return pf_var + entropy

    # def delta(self, *, diffentropy_kw=None):
    #     weights = self._pf.scale_weights().weights
    #     returns = self._pf.as_returns()

    #     diffentropy_kw = {} if diffentropy_kw is None else diffentropy_kw
    #     X_diff_entropy = scipy.stats.differential_entropy(returns)

    #     exp_wH = np.exp(np.sum(weights * X_diff_entropy))

    #     wX_diff_entropy = scipy.stats.differential_entropy(
    #         np.sum(weights * returns)
    #     )

    #     exp_HwX = np.exp(wX_diff_entropy)

    #     ddi = (exp_wH - exp_HwX) / exp_wH

    #     return ddi
