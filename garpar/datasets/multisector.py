# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Multisector class for Garpar project."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

from .ds_base import StocksSetMakerABC
from ..core.stocks_set import StocksSet
from ..utils import Bunch, mabc, unique_names

# =============================================================================
# MULTISECTOR
# =============================================================================


class MultiSector(StocksSetMakerABC):
    """StocksSet maker for creating a multi-sector StocksSet.

    This class allows the user to create a multi-sector StocksSet using
    multiple StocksSet maker objects.
    """

    makers = mabc.hparam(converter=lambda v: tuple(dict(v).items()))

    @makers.validator
    def _makers_validator(self, attribute, value):
        """Validate the StocksSets makers.

        Parameters
        ----------
        attribute : str
            Name of the attribute being validated.
        value : tuple
            Tuple of (maker_name, maker) pairs.

        Raises
        ------
        ValueError
            If there are fewer than 2 makers provided or any maker is not an
            instance of StocksSetMakerABC.
        """
        if len(value) < 2:
            raise ValueError("You must provide at least 2 makers")
        for maker_name, maker in value:
            if not isinstance(maker, StocksSetMakerABC):
                cls_name = StocksSetMakerABC.__name__
                msg = f"Maker '{maker_name}' is not instance of '{cls_name}'"
                raise TypeError(msg)

    def _coerce_price(self, stocks, prices, makers_len):
        """Coerces initial prices into arrays split by the number of makers.

        Parameters
        ----------
        stocks : int
            Number of stocks.
        prices : int, float, or array-like
            Initial prices of stocks.
        makers_len : int
            Number of sector makers.

        Returns
        -------
        numpy.ndarray
            Array of initial prices split by the number of makers.

        Raises
        ------
        ValueError
            If the number of prices does not match the number of stocks.
        """
        if isinstance(prices, (int, float)):
            prices = np.full(stocks, prices, dtype=float)
        elif len(prices) != stocks:
            raise ValueError(f"The number of prices must be equal {stocks}")
        prices = np.asarray(prices, dtype=float)
        return np.array_split(prices, makers_len)

    def make_stocks_set(
        self, *, window_size=5, days=365, stocks=10, price=100, weights=None
    ):
        """Create a multi-sector StocksSet based on specified parameters.

        Parameters
        ----------
        window_size : int, optional
            Window size for StocksSet creation (default is 5).
        days : int, optional
            Number of days for StocksSet evaluation (default is 365).
        stocks : int, optional
            Number of stocks in the StocksSet (default is 10).
        price : int, float, or array-like, optional
            Initial price or prices of stocks (default is 100).
        weights : array-like or None, optional
            Initial weights of stocks (default is None).

        Returns
        -------
        StocksSet
            StocksSet object representing the generated multi-sector stocks
            set.
        """
        makers_len = len(self.makers)
        if stocks < makers_len:
            raise ValueError(f"stocks must be >= {makers_len}")

        prices = self._coerce_price(stocks, price, makers_len)

        stocks_dfs = []
        entropy = []
        metadata = {}
        for (maker_name, maker), maker_prices in zip(self.makers, prices):
            stocks_number = len(maker_prices)
            ss = maker.make_stocks_set(
                window_size=window_size,
                days=days,
                stocks=stocks_number,
                price=maker_prices,
                weights=None,
            )

            df = ss._prices_df.add_prefix(f"{maker_name}_")
            stocks_dfs.append(df)

            entropy.extend(ss.entropy)

            metadata[maker_name] = Bunch(
                maker_name,
                {
                    "stocks": df.columns.to_numpy(),
                    "og_metadata": ss.metadata,
                    "stocks_number": stocks_number,
                },
            )

        # join all the dfs in one
        stock_df = pd.concat(stocks_dfs, axis="columns")

        # create the StocksSet
        return StocksSet.from_prices(
            stock_df,
            weights=weights,
            window_size=window_size,
            entropy=entropy,
            **metadata,
        )


def make_multisector(*makers, **kwargs):
    """Create a multi-sector StocksSet using specified sector makers.

    Parameters
    ----------
    *makers : variable-length arguments
        Instances of StocksSetMakerABC representing sector makers.
    **kwargs : keyword arguments
        Additional parameters passed to MultiSector.make_stocks_set.

    Returns
    -------
    StocksSet
        Multi-sector StocksSet object generated
        by MultiSector.make_stocks_set.
    """
    names = [type(maker).__name__.lower() for maker in makers]
    named_makers = unique_names(names=names, elements=makers)

    ss_maker = MultiSector(named_makers)
    port = ss_maker.make_stocks_set(**kwargs)

    return port
