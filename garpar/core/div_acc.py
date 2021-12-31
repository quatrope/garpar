import attr

import numpy as np

from pypfopt import expected_returns, objective_functions

from . import mixins
from ..utils import aabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class DiversificationAccessor(aabc.AccessorABC, mixins.CoercerMixin):

    _DEFAULT_KIND = "ratio"

    _pf = attr.ib()

    def ratio(self, covariance="sample_cov", covariance_kw=None):
        weights = self._pf.weights
        ret_std = self._pf.as_returns().std()
        pf_variance = self._pf.risk.pf_var(
            covariance=covariance, covariance_kw=covariance_kw
        )

        np.sum(weights * ret_std) / np.sqrt(pf_variance)

    def mrc(self, covariance="sample_cov", covariance_kw=None):
        weights = self._pf.weights

        cov_matrix = self.coerce_covariance_matrix(
            covariance, covariance_kw, asarray=False
        )

        pf_variance = self._pf.risk.pf_var(
            covariance=cov_matrix, covariance_kw=None
        )

        result = np.sum(cov_matrix * weights, axis=1) / np.sqrt(pf_variance)
        result.name = "MRC"
        return result
