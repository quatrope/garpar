import attr

from pypfopt import expected_returns, objective_functions

from..utils import aabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, repr=False, slots=True)
class ReturnsAccessor(aabc.AccessorABC):

    _DEFAULT_KIND = "returns"

    _pf = attr.ib()

    def returns(self, **kwargs):
        return expected_returns.returns_from_prices(
            prices=self._pf._df, **kwargs
        )

    def capm(self, **kwargs):
        returns = expected_returns.capm_return(
            prices=self._pf._df, returns_data=False, **kwargs
        )
        returns.name = "CAPM"
        return returns

    def mah(self, **kwargs):
        returns = expected_returns.mean_historical_return(
            prices=self._pf._df, returns_data=False, **kwargs
        )
        returns.name = "MAH"
        return returns

    def emah(self, **kwargs):
        returns = expected_returns.ema_historical_return(
            prices=self._pf._df, returns_data=False, **kwargs
        )
        returns.name = "EMAH"
        return returns
