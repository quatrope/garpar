# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import inspect

import joblib

import numpy as np

import pandas as pd

from ..core.portfolio import Portfolio
from ..utils import mabc

# =============================================================================
# BASE
# =============================================================================


class PortfolioMakerABC(mabc.ModelABC):
    _MKPORT_SIGNATURE = {
        "self",
        "window_size",
        "days",
        "stocks",
        "price",
        "weights",
    }

    def __init_subclass__(cls):
        mpsig = inspect.signature(cls.make_portfolio)
        missing_args = cls._MKPORT_SIGNATURE.difference(mpsig.parameters)
        if missing_args:
            missing_args_str = ", ".join(missing_args)
            msg = f"Missing arguments {missing_args_str!r} in make_portfolio"
            raise TypeError(msg)
        return super().__init_subclass__()

    @mabc.abstractmethod
    def make_portfolio(
        self,
        *,
        window_size=5,
        days=365,
        stocks=10,
        price=100,
        weights=None,
    ):
        raise NotImplementedError()


# =============================================================================
# ANOTHER BASE
# =============================================================================


class RandomEntropyPortfolioMakerABC(PortfolioMakerABC):
    entropy = mabc.hparam(default=0.5)
    random_state = mabc.hparam(
        default=None, converter=np.random.default_rng, repr=False
    )
    n_jobs = mabc.hparam(default=None)
    verbose = mabc.hparam(default=0)

    # Abstract=================================================================

    @mabc.abstractmethod
    def get_window_loss_probability(self, window_size, entropy):
        raise NotImplementedError()

    @mabc.abstractmethod
    def make_stock_price(self, price, loss, random):
        raise NotImplementedError()

    # INTERNAL ================================================================

    def _coerce_price(self, stocks, prices):
        if isinstance(prices, (int, float)):
            prices = np.full(stocks, prices, dtype=float)
        elif len(prices) != stocks:
            raise ValueError(f"The number of prices must be equal {stocks}")
        return np.asarray(prices, dtype=float)

    def _make_stocks_seeds(self, stocks):
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
        win_probability = 1.0 - loss_probability

        # primero seleccionamos con las probabilidades adecuadas si en cada
        # dia se pierde o se gana
        sequence = random.choice(
            [True, False],
            size=days,
            p=[loss_probability, win_probability],
        )

        # con esto generamos lugares al azar donde vamos a invertir la
        # secuencia anterior dado que las probabilidades representan
        # una distribucion simétrica
        reverse_mask = random.choice([True, False], size=days)

        # finalmente invertimos esos lugares
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
        # determinamos que dia se pierde y que dia se gana
        loss_sequence = self._make_loss_sequence(
            days, loss_probability, random
        )

        # fijamos el primer precio como el precio orginal
        current_price = initial_price

        # vamos a tener tantos dias como dijimos mas un dia 0 al comienzo
        timeserie = np.empty(days + 1, dtype=float)

        # first price is the initial one
        timeserie[0] = current_price

        for day, loss in enumerate(loss_sequence, start=1):
            current_price = self.make_stock_price(current_price, loss, random)
            timeserie[day] = current_price

        stock_df = pd.DataFrame({f"S{stock_idx}": timeserie})

        return stock_df

    # API =====================================================================

    def make_portfolio(
        self,
        *,
        window_size=5,
        days=365,
        stocks=10,
        price=100,
        weights=None,
    ):
        if window_size <= 0:
            raise ValueError("'window_size' must be > 0")
        if days < window_size:
            raise ValueError("'days' must be >= window_size")

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

        return Portfolio.from_dfkws(
            stock_df,
            weights=weights,
            entropy=self.entropy,
            window_size=window_size,
        )
