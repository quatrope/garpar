import attr

from pypfopt import expected_returns, risk_models

from ..utils import aabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class CovarianceAccessor(aabc.AccessorABC):

    _DEFAULT_KIND = "sample_cov"

    _pf = attr.ib()

    def sample_cov(self, **kwargs):
        return risk_models.sample_cov(
            prices=self._pf._df, returns_data=False, **kwargs
        )

    def sample_corr(self, **kwargs):
        cov = self.sample_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def exp_cov(self, **kwargs):
        return risk_models.exp_cov(
            prices=self._pf._df, returns_data=False, **kwargs
        )

    def exp_corr(self, **kwargs):
        cov = self.exp_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def semi_cov(self, **kwargs):
        return risk_models.semicovariance(
            prices=self._pf._df, returns_data=False, **kwargs
        )

    def semi_corr(self, **kwargs):
        cov = self.semi_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def ledoit_wolf_cov(self, shrinkage_target="constant_variance", **kwargs):
        covshrink = risk_models.CovarianceShrinkage(
            prices=self._pf._df, returns_data=False, **kwargs
        )
        return covshrink.ledoit_wolf(shrinkage_target=shrinkage_target)

    def ledoit_wolf_corr(self, **kwargs):
        cov = self.ledoit_wolf_cov(**kwargs)
        return risk_models.cov_to_corr(cov)

    def oracle_approximating_cov(self, **kwargs):
        covshrink = risk_models.CovarianceShrinkage(
            prices=self._pf._df, returns_data=False, **kwargs
        )
        return covshrink.oracle_approximating()

    def oracle_approximating_corr(self, **kwargs):
        cov = self.oracle_approximating_cov(**kwargs)
        return risk_models.cov_to_corr(cov)
