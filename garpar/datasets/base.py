# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================
import abc

import joblib

import numpy as np

import pandas as pd

from . import data
from ..core import portfolio as pf
from ..utils.mabc import ModelABC, hparam, abstractmethod

# =============================================================================
# BASE
# =============================================================================
class PortfolioMakerABC(ModelABC):

    random_state = hparam(
        default=None, converter=np.random.default_rng, repr=False
    )
    n_jobs = hparam(default=None)
    verbose = hparam(default=0)

    # Abstract=================================================================

    @abstractmethod
    def get_window_loss_probability(self, window_size, entropy):
        raise NotImplementedError()

    @abstractmethod
    def make_stock_price(self, price, loss, random):
        raise NotImplementedError()

    # INTERNAL ================================================================

    def _coerce_price(self, stock_number, prices):
        if isinstance(prices, (int, float)):
            prices = np.full(stock_number, prices, dtype=float)
        elif len(prices) != stock_number:
            raise ValueError(f"The q of prices must be equal {stock_number}")
        return np.asarray(prices, dtype=float)

    def _make_stocks_seeds(self, stock_number):
        iinfo = np.iinfo(int)
        seeds = self.random_state.integers(
            low=0,
            high=iinfo.max,
            size=stock_number,
            dtype=iinfo.dtype,
            endpoint=True,
        )

        return seeds

    def _make_loss_sequence(self, days, loss_probability, random):
        win_probability = 1 - loss_probability

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
        entropy=0.5,
        stock_number=10,
        price=100,
        weights=None,
    ):
        if window_size <= 0:
            raise ValueError("'window_size' must be > 0")

        initial_prices = self._coerce_price(stock_number, price)

        loss_probability = self.get_window_loss_probability(
            window_size, entropy
        )

        seeds = self._make_stocks_seeds(stock_number)
        idx_prices_seed = zip(range(stock_number), initial_prices, seeds)

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

        return pf.Portfolio.from_dfkws(
            stock_df,
            weights=weights,
            entropy=entropy,
            window_size=window_size,
        )


def load_merval2021_2022(imputation="ffill"):
    """Argentine stock market prices (MERVAL) from 2020 to early 2022. \
    Unlisted shares were eliminated.

    """
    df = pd.read_csv(data.DATA_PATH / "merval2020-2022.csv", index_col="Days")
    df.index = pd.to_datetime(df.index)

    if imputation in ("backfill", "bfill", "pad", "ffill"):
        df.fillna(method=imputation, inplace=True)
    else:
        df.fillna(value=imputation, inplace=True)

    port = pf.Portfolio.from_dfkws(
        df,
        weights=None,
        title="Merval 2020-2022",
        imputation=imputation,
        description=(
            "Argentine stock market prices (MERVAL) from 2020 to early 2022. "
            "Unlisted shares were eliminated."
        ),
    )

    return port
