# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Juan Cabral, Nadia Luczywo
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

from garpar.makers import base

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_MarketMakerABC_not_implementhed_methods():
    class Foo(base.MarketMakerABC):
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
    class Foo(base.MarketMakerABC):

        faa = base.hparam(default=12)

        def get_window_loss_probability(self, windows_size, entropy):
            ...

        def make_stock_price(self, price, loss, random):
            ...

    maker = Foo()

    assert repr(maker) == "Foo(faa=12)"


def test_MarketMakerABC_bad_coherce_price():
    class Foo(base.MarketMakerABC):
        def get_window_loss_probability(self, windows_size, entropy):
            ...

        def make_stock_price(self, price, loss, random):
            ...

    maker = Foo()

    with pytest.raises(ValueError):
        maker.make_market(price=[100])


@pytest.mark.parametrize(
    "windows_size, sequence",
    [
        (1, [True]),
        (2, [False, True]),
        (3, [False, True, False]),
        (4, [False, True, False, True]),
        (5, [True, False, True, False, True]),
        (6, [True, False, True, False, True, False]),
        (7, [True, False, True, False, True, False, True]),
    ],
)
def test_MarketMakerABC_get_loss_sequence(windows_size, sequence):
    class Foo(base.MarketMakerABC):
        def get_window_loss_probability(self, windows_size, entropy):
            ...

        def make_stock_price(self, price, loss, random):
            ...

    maker = Foo(random_state=10)

    result = maker.get_loss_sequence(
        loss_probability=0.33,
        windows_size=windows_size,
        random=maker.random_state,
    )
    assert np.all(result == sequence)
