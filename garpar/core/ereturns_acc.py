# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import attr

from pypfopt import expected_returns

from ..utils import accabc

# =============================================================================
#
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class ExpectedReturnsAccessor(accabc.AccessorABC):
    _default_kind = "capm"

    _pf = attr.ib()

    def capm(self, **kwargs):
        returns = expected_returns.capm_return(
            prices=self._pf._prices_df, returns_data=False, **kwargs
        )
        returns.name = "CAPM"
        return returns

    def mah(self, **kwargs):
        returns = expected_returns.mean_historical_return(
            prices=self._pf._prices_df, returns_data=False, **kwargs
        )
        returns.name = "MAH"
        return returns

    def emah(self, **kwargs):
        returns = expected_returns.ema_historical_return(
            prices=self._pf._prices_df, returns_data=False, **kwargs
        )
        returns.name = "EMAH"
        return returns
