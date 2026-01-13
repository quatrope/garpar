# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2026
#   Lautaro Ebner,
#   Diego Gimenez,
#   Nadia Luczywo,
#   Juan Cabral,
#   and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Expected Returns Accessor.

The expected returns accessor module provides an accessor class with methods to
compute expected returns for a given stocks set.

Key Features:
    - Expected returns estimation

Example
-------
    >>> import garpar
    >>> ss = garpar.mkss(prices=[...])
    >>> ss.expected_returns.capm()
    >>> ss.expected_returns.mah()
    >>> ss.expected_returns.emah()

"""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

from pypfopt import expected_returns

from ..utils import AccessorABC

# =============================================================================
# EXPECTED RETURNS
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class ExpectedReturnsAccessor(AccessorABC):
    """Accessor class for computing expected returns of a stocks set."""

    _default_kind = "capm"

    _ss = attr.ib()

    def capm(self, **kwargs):
        """Compute expected returns using the CAPM method.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            expected_returns.capm_return.

        Returns
        -------
        pandas.Series
            Series containing computed expected returns with name "CAPM".
        """
        returns = expected_returns.capm_return(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )
        returns.name = "CAPM"
        return returns

    def mah(self, **kwargs):
        """Compute expected returns using the mean historical method.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            expected_returns.mean_historical_return.

        Returns
        -------
        pandas.Series
            Series containing computed expected returns with name "MAH".
        """
        returns = expected_returns.mean_historical_return(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )
        returns.name = "MAH"
        return returns

    def emah(self, **kwargs):
        """Compute expected returns using the emah method.

        Compute expected returns using the
        exponential moving average historical method.

        Parameters
        ----------
        **kwargs : keyword arguments
            Additional keyword arguments passed to
            `expected_returns.ema_historical_return`. For example, you might
            pass parameters like `span`, `min_periods`, etc., which control
            the behavior of the exponential moving average calculation.

        Returns
        -------
        pandas.Series
            A pandas Series containing the computed expected returns using the
            exponential moving average historical method, with the name
            "EMAH".
        """
        returns = expected_returns.ema_historical_return(
            prices=self._ss._prices_df, returns_data=False, **kwargs
        )
        returns.name = "EMAH"
        return returns
