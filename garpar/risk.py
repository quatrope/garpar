import attr

from pypfopt import expected_returns
from pypfopt import risk_models


@attr.s(frozen=True, repr=False, slots=True)
class RiskAccessor:

    _port = attr.ib()

    def mean_historical_return(self, **kwargs):
        portfolio = self._port
        return expected_returns.mean_historical_return(
            prices=portfolio._df, **kwargs
        )

    def sample_covariance(self, **kwargs):
        portfolio = self._port
        return risk_models.sample_cov(prices=portfolio._df, **kwargs)
