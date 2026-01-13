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

"""Test Datasets base module."""

# =============================================================================
# IMPORTS
# =============================================================================

from garpar import datasets

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_StocksSetMakerABC_make_stocks_set_missing_argument_on_definition():
    """Test StockstSetMakerABC make_stocks_set missing on def."""
    with pytest.raises(TypeError):

        class StocksSetMaker(datasets.StocksSetMakerABC):
            def make_stocks_set(self):
                pass


def test_StocksSetMakerABC_make_stocks_set_missing_not_imp():
    """Test StockstSetMakerABC make_stocks_set missing implementation."""

    class StocksSetMaker(datasets.StocksSetMakerABC):
        def make_stocks_set(
            self,
            *,
            window_size=5,
            days=365,
            stocks=10,
            price=100,
            weights=None,
        ):
            return super().make_stocks_set(
                window_size=window_size,
                days=days,
                stocks=stocks,
                price=price,
                weights=weights,
            )

    maker = StocksSetMaker()
    with pytest.raises(NotImplementedError):
        maker.make_stocks_set()


def test_RandomEntropyPorfolioMakerABC_get_window_loss_probability_not_imp():
    """Test StockstSetMakerABC make_stocks_set get_window_loss no imp."""

    class StocksSetMaker(datasets.RandomEntropyStocksSetMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return super().get_window_loss_probability(window_size, entropy)

        def make_stock_price(self, price, loss, random):
            return price

    maker = StocksSetMaker()

    with pytest.raises(NotImplementedError):
        maker.make_stocks_set()


def test_RandomEntropyPorfolioMakerABC_make_stock_price_not_imp():
    """Test StockstSetMakerABC make_stock_price not implemented."""

    class StocksSetMaker(datasets.RandomEntropyStocksSetMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return super().make_stock_price(price, loss, random)

    maker = StocksSetMaker()

    with pytest.raises(NotImplementedError):
        maker.make_stocks_set()


def test_RandomEntropyPorfolioMakerABC_window_size_le_0():
    """Test StockstSetMakerABC windows size less than 0."""

    class StocksSetMaker(datasets.RandomEntropyStocksSetMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = StocksSetMaker()

    with pytest.raises(ValueError):
        maker.make_stocks_set(window_size=0)


def test_RandomEntropyPorfolioMakerABC_window_days_lt_window_size():
    """Test RandomEntropyPorfolioMakerABC window days less than window size."""

    class StocksSetMaker(datasets.RandomEntropyStocksSetMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = StocksSetMaker()

    with pytest.raises(ValueError):
        maker.make_stocks_set(window_size=5, days=4)


def test_RandomEntropyPorfolioMakerABC_invalid_number_of_prices():
    """Test RandomEntropyPorfolioMakerABC invalid number of prices."""

    class StocksSetMaker(datasets.RandomEntropyStocksSetMakerABC):
        def get_window_loss_probability(self, window_size, entropy):
            return 0.2

        def make_stock_price(self, price, loss, random):
            return price

    maker = StocksSetMaker()

    with pytest.raises(ValueError):
        maker.make_stocks_set(stocks=2, price=[1])
