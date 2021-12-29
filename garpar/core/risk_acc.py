import attr

from pypfopt import expected_returns

from ..utils import aabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, repr=False, slots=True)
class RiskAccessor(aabc.AccessorABC):

    _DEFAULT_KIND = "beta"

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
