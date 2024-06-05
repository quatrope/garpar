# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar import datasets

import numpy as np

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_PortfolioMakerABC_make_portfolio_missing_argument_on_definition():
    with pytest.raises(TypeError):

        class PortfolioMaker(datasets.PortfolioMakerABC):
            def make_portfolio(self):
                pass


def test_PortfolioMakerABC_make_portfolio_missing_not_imp():
    class PortfolioMaker(datasets.PortfolioMakerABC):
        def make_portfolio(
            self,
            *,
            window_size=5,
            days=365,
            stocks=10,
            price=100,
            weights=None,
        ):
            return super().make_portfolio(
                window_size=window_size,
                days=days,
                stocks=stocks,
                price=price,
                weights=weights,
            )

    maker = PortfolioMaker()
    with pytest.raises(NotImplementedError):
        maker.make_portfolio()


def test_RandomEntropyPorfolioMakerABC_get_window_loss_probability_not_imp():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return super().get_window_loss_probability(window_size, entropy)

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(NotImplementedError):
        maker.make_portfolio()


def test_RandomEntropyPorfolioMakerABC_make_stock_price_not_imp():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return super().make_stock_price(price, loss, random)

    maker = PortfolioMaker()

    with pytest.raises(NotImplementedError):
        maker.make_portfolio()


def test_RandomEntropyPorfolioMakerABC_window_size_le_0():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(ValueError):
        maker.make_portfolio(window_size=0)


def test_RandomEntropyPorfolioMakerABC_window_days_lt_window_size():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(ValueError):
        maker.make_portfolio(window_size=5, days=4)


def test_RandomEntropyPorfolioMakerABC_invalid_number_of_prices():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(ValueError):
        maker.make_portfolio(stocks=2, price=[1])
