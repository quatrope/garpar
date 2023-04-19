import attr

import numpy as np

from pypfopt import risk_models

from ..utils import accabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class CovarianceAccessor(accabc.AccessorABC):
    _default_kind = "sample_cov"

    _pf = attr.ib()

    def sample_cov(self, **kwargs):
        return risk_models.sample_cov(
            prices=self._pf._df, returns_data=False, **kwargs
        )

    def exp_cov(self, **kwargs):
        return risk_models.exp_cov(
            prices=self._pf._df, returns_data=False, **kwargs
        )

    def semi_cov(self, **kwargs):
        return risk_models.semicovariance(
            prices=self._pf._df, returns_data=False, **kwargs
        )

    def ledoit_wolf_cov(self, shrinkage_target="constant_variance", **kwargs):
        covshrink = risk_models.CovarianceShrinkage(
            prices=self._pf._df, returns_data=False, **kwargs
        )
        return covshrink.ledoit_wolf(shrinkage_target=shrinkage_target)

    def oracle_approximating_cov(self, **kwargs):
        covshrink = risk_models.CovarianceShrinkage(
            prices=self._pf._df, returns_data=False, **kwargs
        )
        return covshrink.oracle_approximating()


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class CorrelationAccessor(accabc.AccessorABC):
    _default_kind = "sample_corr"

    _pf = attr.ib()

    def sample_corr(self, **kwargs):
        cov = self._pf.covariance.sample_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def exp_corr(self, **kwargs):
        cov = self._pf.covariance.exp_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def semi_corr(self, **kwargs):
        cov = self._pf.covariance.semi_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def ledoit_wolf_corr(self, **kwargs):
        cov = self._pf.covariance.ledoit_wolf_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def oracle_approximating_corr(self, **kwargs):
        cov = self._pf.covariance.oracle_approximating_cov(**kwargs)
        return risk_models.cov_to_corr(cov)
