import attr

import numpy as np

import scipy.stats

from sklearn.decomposition import PCA

from . import _mixins
from ..utils import accabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class DiversificationAccessor(accabc.AccessorABC, _mixins.CoercerMixin):
    _default_kind = "ratio"

    _pf = attr.ib()

    def ratio(self, *, covariance="sample_cov", covariance_kw=None):
        """Diversification ratio."""
        weights = self._pf.scale_weights().weights
        ret_std = self._pf.as_returns().std()
        pf_variance = self._pf.risk.pf_var(
            covariance=covariance, covariance_kw=covariance_kw
        )

        return np.sum(weights * ret_std) / np.sqrt(pf_variance)

    def mrc(self, *, covariance="sample_cov", covariance_kw=None):
        """Marginal risk contribution"""
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
        """Portfolio diversification index."""

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
        weights = self._pf.scale_weights().weights
        return -np.sum(weights * np.log(weights))

    def cross_entropy(self, benchmark_weights=None):
        """"""
        weights = self._pf.scale_weights().weights
        benchmark_weights = self.coerce_weights(benchmark_weights)
        return np.sum(benchmark_weights * np.log(benchmark_weights / weights))

    def ke_zang_entropy(self, *, covariance="sample_cov", covariance_kw=None):
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
