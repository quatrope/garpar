import attr

from pypfopt import expected_returns, risk_models

from ..utils import aabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, repr=False, slots=True)
class RiskAccessor(aabc.AccessorABC):

    _DEFAULT_KIND = "sample_cov"

    _pf = attr.ib()

    def beta(
        self,
        market_prices=None,
        log_returns=False,
    ):

        prices = self._pf._df
        market_returns = None

        returns = expected_returns.returns_from_prices(prices, log_returns)
        if market_prices is not None:
            market_returns = expected_returns.returns_from_prices(
                market_prices, log_returns
            )

        # Use the equally-weighted dataset as a proxy for the market
        if market_returns is None:
            # Append market return to right and compute sample covariance matrix
            returns["mkt"] = returns.mean(axis=1)
        else:
            market_returns.columns = ["mkt"]
            returns = returns.join(market_returns, how="left")

        # Compute covariance matrix for the new dataframe (including markets)
        cov = returns.cov()
        # The far-right column of the cov matrix is covariances to market
        betas = cov["mkt"] / cov.loc["mkt", "mkt"]
        betas = betas.drop("mkt")
        betas.name = "beta"

        return betas

    # pyopt.risk_models
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