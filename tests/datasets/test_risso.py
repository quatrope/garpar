# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test Risso Module."""

# =============================================================================
# IMPORTS
# =============================================================================


from garpar import StocksSet, datasets

import numpy as np

import pandas as pd


# =============================================================================
# TESTS
# =============================================================================


def test_argnearest():
    """Test argnearest function."""
    assert datasets.risso.argnearest([0.1, 0.11, -0.001], 0) == 2


# =============================================================================
# UNIFORM
# =============================================================================


def test_RissoUniform_delisted():
    """Test RissoUniform delisted."""
    maker = datasets.RissoUniform()
    assert maker.make_stock_price(0, True, np.random.default_rng()) == 0


def test_make_risso_uniform():
    """Test make_risso_uniform."""
    result = datasets.make_risso_uniform(
        random_state=42, window_size=2, days=3, stocks=2
    )
    expected = StocksSet.from_prices(
        pd.DataFrame(
            [
                [100.000000, 100.000000],
                [103.992997, 95.595250],
                [99.558047, 92.252368],
                [102.419657, 93.418097],
            ],
            columns=["S0", "S1"],
        ),
        weights=[1, 1],
    )

    pd.testing.assert_frame_equal(result.as_prices(), expected.as_prices())
    pd.testing.assert_series_equal(result.weights, expected.weights)


# =============================================================================
# NORMAL
# =============================================================================


def test_RissoNormal_delisted():
    """Test RissoNormal delisted."""
    maker = datasets.RissoNormal()
    assert maker.make_stock_price(0, True, np.random.default_rng()) == 0


def test_make_risso_normal():
    """Test make_risso_normal."""
    result = datasets.make_risso_normal(
        random_state=42, window_size=2, days=3, stocks=2
    )
    expected = StocksSet.from_prices(
        pd.DataFrame(
            [
                [100.000000, 100.000000],
                [100.848584, 99.265290],
                [99.946433, 98.476856],
                [100.794898, 99.764455],
            ],
            columns=["S0", "S1"],
        ),
        weights=[1, 1],
    )

    pd.testing.assert_frame_equal(result.as_prices(), expected.as_prices())
    pd.testing.assert_series_equal(result.weights, expected.weights)


# =============================================================================
# LEVY STABLE
# =============================================================================


def test_RissoLevyStable_delisted():
    """Test RissoLevyStable delisted."""
    maker = datasets.RissoLevyStable()
    assert maker.make_stock_price(0, True, np.random.default_rng()) == 0


def test_make_risso_levy_stable():
    """Test make_risso_levy_stable."""
    result = datasets.make_risso_levy_stable(
        random_state=42, window_size=2, days=3, stocks=2
    )
    expected = StocksSet.from_prices(
        pd.DataFrame(
            [
                [100.0, 100.0],
                [100.240114, 99.916195],
                [100.160582, 99.828921],
                [101.422445, 100.399826],
            ],
            columns=["S0", "S1"],
        ),
        weights=[1, 1],
    )

    pd.testing.assert_frame_equal(
        result.as_prices(),
        expected.as_prices(),
        check_exact=False,
        atol=0.01,
        rtol=0.01,
    )
    pd.testing.assert_series_equal(result.weights, expected.weights)
