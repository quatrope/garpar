# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Juan Cabral, Nadia Luczywo
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

from io import StringIO

from garpar.datasets import risso

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_risso_argnearest():
    assert risso.argnearest([0.1, -0.98], 0) == 0
    assert risso.argnearest([0.1, -0.98], -0.99) == 1
    assert risso.argnearest([0.1, -0.10], 0.1) == 0


@pytest.mark.parametrize(
    "windows_size, h",
    [
        (1, [0.0, 0.0]),
        (2, [0.0, 1.0, 0.0]),
        (3, [0.0, 0.91830, 0.91830, 0.0]),
        (4, [0.0, 0.81127, 1.0, 0.81127, 0.0]),
        (5, [0.0, 0.72193, 0.97095, 0.97095, 0.72193, 0.0]),
        (6, [0.0, 0.65002, 0.91830, 1.0, 0.91830, 0.65002, 0.0]),
        (7, [0.0, 0.59167, 0.86312, 0.98522, 0.98522, 0.86312, 0.59167, 0.0]),
    ],
)
def test_RissoNormal_candidate_entropy(windows_size, h):
    maker = risso.RissoNormal()
    me, _ = maker.risso_candidate_entropy(windows_size)
    assert np.allclose(me, h, atol=1e-05)


@pytest.mark.parametrize("windows_size", [0, -1])
def test_RissoNormal_candidate_entropy_le0(windows_size):
    maker = risso.RissoNormal()
    with pytest.raises(ValueError):
        maker.risso_candidate_entropy(windows_size)


def test_RissoNormal_make_stock_price():
    maker = risso.RissoNormal()
    assert maker.make_stock_price(100, True, maker.random_state) < 100
    assert maker.make_stock_price(100, False, maker.random_state) > 100
    assert maker.make_stock_price(0, False, maker.random_state) == 0
    assert maker.make_stock_price(0, True, maker.random_state) == 0


@pytest.mark.parametrize(
    "windows_size, loss_prob",
    [
        (1, 0.0),
        (2, 0.0),
        (3, 0.33333),
        (4, 0.25000),
        (5, 0.20000),
        (6, 0.16667),
        (7, 0.14286),
    ],
)
def test_RissoNormal_window_loss_probability(windows_size, loss_prob):
    maker = risso.RissoNormal()
    result = maker.get_window_loss_probability(windows_size, 0.5)
    assert np.allclose(result, loss_prob, atol=1e-05)


def test_RissoNormal_make_market():
    csv_code = """
        stock_0_price,stock_1_price
        100.859292,99.256746
        100.490542,98.713592
        99.531659,99.379101
        100.410109,99.611263
        100.360183,99.727948
        100.545046,99.509260
        99.864116,98.637831
        101.086658,98.414235
        100.932128,99.093149
        100.503800,99.160728
    """

    csv_code = "\n".join([line.strip() for line in csv_code.splitlines()])
    expected = pd.read_csv(StringIO(csv_code))

    maker = risso.RissoNormal(random_state=42, mu=0.0, sigma=1.0)
    result = maker.make_market(
        days=10, window_size=5, stock_number=2, entropy=0.5
    )

    assert result.entropy == 0.5
    assert result.window_size == 5
    assert len(result) == 10
    assert np.all(result.initial_prices == [100, 100])
    pd.testing.assert_frame_equal(result._df, expected, atol=1e-10)
