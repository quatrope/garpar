# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Expected Returns Accessor."""

import attr

from pypfopt import expected_returns

from ..utils import accabc

# =============================================================================
# EXPECTED RETURNS
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class ExpectedReturnsAccessor(accabc.AccessorABC):
    """Accessor class for computing expected returns of a portfolio.

    Attributes
    ----------
    _default_kind : str
        Default method for computing expected returns.
    _pf : Portfolio
        Portfolio object containing prices data.

    Methods
    -------
    capm(**kwargs)
        Compute expected returns using the CAPM method.
    mah(**kwargs)
        Compute expected returns using the mean historical method.
    emah(**kwargs)
        Compute expected returns using the exponential moving average historical method.
    """

    _default_kind = "capm"

    _pf = attr.ib()

    def capm(self, **kwargs):
        """Compute expected returns using the CAPM (Capital Asset Pricing Model) method.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to expected_returns.capm_return.

        Returns
        -------
        pandas.Series
            Series containing computed expected returns with name "CAPM".
        """
        returns = expected_returns.capm_return(
            prices=self._pf._prices_df, returns_data=False, **kwargs
        )
        returns.name = "CAPM"
        return returns

    def mah(self, **kwargs):
        """Compute expected returns using the mean historical method.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to expected_returns.mean_historical_return.

        Returns
        -------
        pandas.Series
            Series containing computed expected returns with name "MAH".
        """
        returns = expected_returns.mean_historical_return(
            prices=self._pf._prices_df, returns_data=False, **kwargs
        )
        returns.name = "MAH"
        return returns

    def emah(self, **kwargs):
        """Compute expected returns using the exponential moving average historical method.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to expected_returns.ema_historical_return.

        Returns
        -------
        pandas.Series
            Series containing computed expected returns with name "EMAH".
        """
        returns = expected_returns.ema_historical_return(
            prices=self._pf._prices_df, returns_data=False, **kwargs
        )
        returns.name = "EMAH"
        return returns
