# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
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


def test_PorfolioMakerABC_get_window_loss_probability_not_implemethed():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return super().get_window_loss_probability(window_size, entropy)

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(NotImplementedError):
        maker.make_portfolio()


def test_PorfolioMakerABC_make_stock_price_not_implemethed():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return super().make_stock_price(price, loss, random)

    maker = PortfolioMaker()

    with pytest.raises(NotImplementedError):
        maker.make_portfolio()


def test_PorfolioMakerABC_window_size_le_0():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(ValueError):
        maker.make_portfolio(window_size=0)


def test_PorfolioMakerABC_window_days_lt_window_size():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(ValueError):
        maker.make_portfolio(window_size=5, days=4)


def test_PorfolioMakerABC_invalid_number_of_prices():
    class PortfolioMaker(datasets.RandomEntropyPortfolioMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = PortfolioMaker()

    with pytest.raises(ValueError):
        maker.make_portfolio(stocks=2, price=[1])
