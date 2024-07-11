# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Multisector."""

import numpy as np

import pandas as pd

from .base import PortfolioMakerABC
from ..core.portfolio import Portfolio
from ..utils import Bunch, mabc, unique_names


class MultiSector(PortfolioMakerABC):
    """Portfolio maker for creating a multi-sector portfolio.

    Attributes
    ----------
    makers : tuple
        Tuple of (maker_name, PortfolioMakerABC) pairs representing different sector makers.

    Methods
    -------
    make_portfolio(*, window_size=5, days=365, stocks=10, price=100, weights=None)
        Creates a multi-sector portfolio based on specified parameters.

    Notes
    -----
    This class extends PortfolioMakerABC and allows for creating a portfolio with
    multiple sectors, each handled by a different PortfolioMakerABC instance.
    """

    makers = mabc.hparam(converter=lambda v: tuple(dict(v).items()))

    @makers.validator
    def _makers_validator(self, attribute, value):
        """Validate 'makers' attribute.

        Parameters
        ----------
        attribute : str
            Name of the attribute being validated.
        value : tuple
            Tuple of (maker_name, maker) pairs.

        Raises
        ------
        ValueError
            If there are fewer than 2 makers provided or any maker is not an instance of PortfolioMakerABC.
        """
        if len(value) < 2:
            raise ValueError("You must provide at least 2 makers")
        for maker_name, maker in value:
            if not isinstance(maker, PortfolioMakerABC):
                cls_name = PortfolioMakerABC.__name__
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

    def make_portfolio(
        self, *, window_size=5, days=365, stocks=10, price=100, weights=None
    ):
        """Create a multi-sector portfolio based on specified parameters.

        Parameters
        ----------
        window_size : int, optional
            Window size for portfolio creation (default is 5).
        days : int, optional
            Number of days for portfolio evaluation (default is 365).
        stocks : int, optional
            Number of stocks in the portfolio (default is 10).
        price : int, float, or array-like, optional
            Initial price or prices of stocks (default is 100).
        weights : array-like or None, optional
            Initial weights of stocks (default is None).

        Returns
        -------
        Portfolio
            Portfolio object representing the generated multi-sector portfolio.
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
            pf = maker.make_portfolio(
                window_size=window_size,
                days=days,
                stocks=stocks_number,
                price=maker_prices,
                weights=None,
            )

            df = pf._prices_df.add_prefix(f"{maker_name}_")
            stocks_dfs.append(df)

            entropy.extend(pf.entropy)

            metadata[maker_name] = Bunch(
                maker_name,
                {
                    "stocks": df.columns.to_numpy(),
                    "og_metadata": pf.metadata,
                    "stocks_number": stocks_number,
                },
            )

        # join all the dfs in one
        stock_df = pd.concat(stocks_dfs, axis="columns")

        # create the portfolio
        return Portfolio.from_dfkws(
            stock_df,
            weights=weights,
            window_size=window_size,
            entropy=entropy,
            **metadata,
        )


def make_multisector(*makers, **kwargs):
    """Create a multi-sector portfolio using specified sector makers.

    Parameters
    ----------
    *makers : variable-length arguments
        Instances of PortfolioMakerABC representing sector makers.
    **kwargs : keyword arguments
        Additional parameters passed to MultiSector.make_portfolio.

    Returns
    -------
    Portfolio
        Multi-sector portfolio object generated by MultiSector.make_portfolio.

    Notes
    -----
    This function creates a multi-sector portfolio by initializing a MultiSector
    object with unique names for each sector maker and then calling make_portfolio
    with specified parameters.

    Example
    -------
    Example usage:

    >>> from mymodule import make_multisector, CustomSectorMaker1, CustomSectorMaker2

    >>> port = make_multisector(
    >>>     CustomSectorMaker1(),
    >>>     CustomSectorMaker2(),
    >>>     window_size=7,
    >>>     days=250,
    >>>     stocks=15,
    >>>     price=200,
    >>>     weights=[0.2, 0.3, 0.5]
    >>> )
    """
    names = [type(maker).__name__.lower() for maker in makers]
    named_makers = unique_names(names=names, elements=makers)

    pf_maker = MultiSector(named_makers)
    port = pf_maker.make_portfolio(**kwargs)

    return port
