# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Abstract classes for StocksSet makers.

This module provides abstract base classes for StocksSet makers.
StocksSetsMakerABC is the base class for all StocksSet makers. While
RandomEntropyStocksSetMakerABC is the base class for all random entropy-based
StocksSet makers.

Key Features:
    - Entropy-based portfolio simulation

"""

# =============================================================================
# IMPORTS
# =============================================================================

import inspect

import joblib

import numpy as np

import pandas as pd

from ..core.stocks_set import StocksSet
from ..utils import mabc

# =============================================================================
# BASE
# =============================================================================


class StocksSetMakerABC(mabc.ModelABC):
    """Abstract base class for defining a stocks set maker.

    This class provides a base for all stocks set makers in the project and
    ensures that all subclasses implement a make_stocks_set method.

    """

    _MKSS_SIGNATURE = {
        "self",
        "window_size",
        "days",
        "stocks",
        "price",
        "weights",
    }

    def __init_subclass__(cls):
        """Ensure that the make_stocks_set method in subclasses fits criteria.

        Ensure that the make_stocks_set method in subclasses
        conforms to _MKSS_SIGNATURE.

        Raises
        ------
        TypeError
            If make_stocks_set method signature does not match _MKSS_SIGNATURE.
        """
        mpsig = inspect.signature(cls.make_stocks_set)
        missing_args = cls._MKSS_SIGNATURE.difference(mpsig.parameters)
        if missing_args:
            missing_args_str = ", ".join(missing_args)
            msg = f"Missing arguments {missing_args_str!r} in make_stocks_set"
            raise TypeError(msg)
        return super().__init_subclass__()

    @mabc.abstractmethod
    def make_stocks_set(
        self,
        *,
        window_size=5,
        days=365,
        stocks=10,
        price=100,
        weights=None,
    ):
        """Abstract factory method to create a stocks set.

        Parameters
        ----------
        window_size : int, optional
            Window size for stocks set creation (default is 5).
        days : int, optional
            Number of days for stocks set evaluation (default is 365).
        stocks : int, optional
            Number of stocks in the stocks set (default is 10).
        price : float, optional
            Initial price for stocks (default is 100).
        weights : array-like or None, optional
            Initial weights of stocks (default is None).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


# =============================================================================
# ANOTHER BASE
# =============================================================================


class RandomEntropyStocksSetMakerABC(StocksSetMakerABC):
    """Abstract class for generating StocksSets with entropy-based prices."""

    # ATTRIBUTES =============================================================

    entropy = mabc.hparam(default=0.5)
    random_state = mabc.hparam(
        default=None, converter=np.random.default_rng, repr=False
    )
    n_jobs = mabc.hparam(default=None)
    verbose = mabc.hparam(default=0)

    # ABSTRACT ==============================================================

    @mabc.abstractmethod
    def get_window_loss_probability(self, window_size, entropy):
        """Abstract method to calculate the loss probability by window size.

        Parameters
        ----------
        window_size : int
            Window size for StocksSet creation.
        entropy : float
            Entropy parameter for StocksSet creation.

        Returns
        -------
        float
            Loss probability for the given window size and entropy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()

    @mabc.abstractmethod
    def make_stock_price(self, price, loss, random):
        """Abstract method to calculate the stock price.

        Parameters
        ----------
        price : float
            Initial price of the stock.
        loss : bool
            Whether there is a loss on the current day.
        random : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Updated stock price based on loss and randomness.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()

    # INTERNAL ================================================================

    def _coerce_price(self, stocks, prices):
        """Coerces the initial prices into an array of float values.

        Parameters
        ----------
        stocks : int
            Number of stocks.
        prices : int, float, or array-like
            Initial prices of stocks.

        Returns
        -------
        numpy.ndarray
            Array of initial prices.

        Raises
        ------
        ValueError
            If the number of prices does not match the number of stocks.
        """
        if isinstance(prices, (int, float)):
            prices = np.full(stocks, prices, dtype=float)
        elif len(prices) != stocks:
            raise ValueError(f"The number of prices must be equal {stocks}")
        return np.asarray(prices, dtype=float)

    def _make_stocks_seeds(self, stocks):
        """Generate seeds for random number generation for each stock.

        Parameters
        ----------
        stocks : int
            Number of stocks.

        Returns
        -------
        numpy.ndarray
            Array of seeds for random number generation.
        """
        iinfo = np.iinfo(int)
        seeds = self.random_state.integers(
            low=0,
            high=iinfo.max,
            size=stocks,
            dtype=iinfo.dtype,
            endpoint=True,
        )

        return seeds

    def _make_loss_sequence(self, days, loss_probability, random):
        """Generate a sequence of losses based on the given loss probability.

        Parameters
        ----------
        days : int
            Number of days.
        loss_probability : float
            Probability of loss on each day.
        random : numpy.random.Generator
            Random number generator.

        Returns
        -------
        numpy.ndarray
            Boolean array indicating loss (True) or no loss (False) on each
            day.
        """
        win_probability = 1.0 - loss_probability

        # select based on loss_probability the raises and drops of an asset
        sequence = random.choice(
            [True, False],
            size=days,
            p=[loss_probability, win_probability],
        )

        reverse_mask = random.choice([True, False], size=days)

        final_sequence = np.where(reverse_mask, ~sequence, sequence)

        return final_sequence

    def _make_stock(
        self,
        days,
        loss_probability,
        stock_idx,
        initial_price,
        random,
    ):
        """Generate a DataFrame for a single stock with random prices.

        Parameters
        ----------
        days : int
            Number of days.
        loss_probability : float
            Probability of loss on each day.
        stock_idx : int
            Index of the stock.
        initial_price : float
            Initial price of the stock.
        random : numpy.random.Generator
            Random number generator.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the stock prices for each day.
        """
        loss_sequence = self._make_loss_sequence(
            days, loss_probability, random
        )

        current_price = initial_price

        timeserie = np.empty(days + 1, dtype=float)

        # first price is the initial one
        timeserie[0] = current_price

        for day, loss in enumerate(loss_sequence, start=1):
            current_price = self.make_stock_price(current_price, loss, random)
            timeserie[day] = current_price

        stock_df = pd.DataFrame({f"S{stock_idx}": timeserie})

        return stock_df

    # API =====================================================================

    def make_stocks_set(
        self,
        *,
        window_size=5,
        days=365,
        stocks=10,
        price=100,
        weights=None,
    ):
        """Create a StocksSet instance with random prices.

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
        garpar.core.stocks_set.StocksSet
            Object representing the generated stocks prices
            along multiple days.

        """
        if window_size <= 0 or days < window_size:
            raise ValueError("'window_size' must be > 0")

        initial_prices = self._coerce_price(stocks, price)

        loss_probability = self.get_window_loss_probability(
            window_size, self.entropy
        )

        seeds = self._make_stocks_seeds(stocks)
        idx_prices_seed = zip(range(stocks), initial_prices, seeds)

        with joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="processes",
        ) as P:
            dmaker = joblib.delayed(self._make_stock)
            stocks = P(
                dmaker(
                    days=days,
                    loss_probability=loss_probability,
                    stock_idx=stock_idx,
                    initial_price=stock_price,
                    random=np.random.default_rng(seed),
                )
                for stock_idx, stock_price, seed in idx_prices_seed
            )

        stock_df = pd.concat(stocks, axis=1)

        return StocksSet.from_prices(
            stock_df,
            weights=weights,
            entropy=self.entropy,
            window_size=window_size,
        )
