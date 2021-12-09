# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

from garpar.datasets import base

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_MarketMakerABC_not_implementhed_methods():
    class Foo(base.PortfolioMakerABC):
        def get_window_loss_probability(self, windows_size, entropy):
            super().get_window_loss_probability(windows_size, entropy)

        def make_stock_price(self, price, loss, random):
            super().make_stock_price(price, loss, random)

    maker = Foo()

    with pytest.raises(NotImplementedError):
        maker.get_window_loss_probability(7, 0.5)

    with pytest.raises(NotImplementedError):
        maker.make_stock_price(0, True, maker.random_state)


def test_MarketMakerABC_repr():
    class Foo(base.PortfolioMakerABC):

        faa = base.hparam(default=12)

        def get_window_loss_probability(self, windows_size, entropy):
            ...

        def make_stock_price(self, price, loss, random):
            ...

    maker = Foo()

    assert repr(maker) == "Foo(faa=12, n_jobs=None, verbose=0)"


def test_MarketMakerABC_bad_coherce_price():
    class Foo(base.PortfolioMakerABC):
        def get_window_loss_probability(self, windows_size, entropy):
            ...

        def make_stock_price(self, price, loss, random):
            ...

    maker = Foo()

    with pytest.raises(ValueError):
        maker.make_portfolio(price=[100])


@pytest.mark.parametrize(
    "days, sequence",
    [
        (1, [True]),
        (2, [False, True]),
        (3, [False, False, False]),
        (4, [False, True, True, False]),
        (5, [True, False, True, True, True]),
        (6, [True, True, True, True, True, False]),
        (7, [True, True, True, False, False, True, True]),
    ],
)
def test_MarketMakerABC_get_loss_sequence(days, sequence):
    class Foo(base.PortfolioMakerABC):
        def get_window_loss_probability(self, windows_size, entropy):
            ...

        def make_stock_price(self, price, loss, random):
            ...

    maker = Foo(random_state=10)

    result = maker.get_loss_sequence(
        loss_probability=0.33,
        days=days,
        random=maker.random_state,
    )

    assert len(result) == days
    assert np.all(result == sequence)
