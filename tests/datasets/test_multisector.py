# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test Multisector module."""

# =============================================================================
# IMPORTS
# =============================================================================


from garpar import StocksSet, datasets

import pandas as pd

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_make_multisector():
    """Test make_multisector."""
    result = datasets.make_multisector(
        datasets.RissoUniform(random_state=42),
        datasets.RissoNormal(random_state=42),
        window_size=2,
        days=3,
        stocks=2,
    )

    expected = StocksSet.from_prices(
        [
            [100.0, 100.0],
            [103.99299697386559, 100.15141604581204],
            [99.72833455414923, 100.04597610435215],
            [102.60264776761953, 100.19705644445486],
        ],
        stocks=["rissouniform_S0", "rissonormal_S0"],
        weights=[1, 1],
    )

    pd.testing.assert_frame_equal(result.as_prices(), expected.as_prices())
    pd.testing.assert_series_equal(result.weights, expected.weights)


def test_make_multisector_to_few_maker():
    """Test make_multisector with few maker."""
    with pytest.raises(ValueError):
        datasets.make_multisector(datasets.RissoUniform())


def test_make_multisector_maker_is_not_instance_of_StocksSetMakerABC():
    """Test make_multisector maker is not instance of StocksSetMakerABC."""
    with pytest.raises(TypeError):
        datasets.make_multisector(datasets.RissoUniform(), None)


def test_make_multisector_prices_and_stocks_different_length():
    """Test make_multisector prices and stocks different length."""
    with pytest.raises(ValueError):
        datasets.make_multisector(
            datasets.RissoUniform(random_state=42),
            datasets.RissoNormal(random_state=42),
            stocks=2,
            price=[1, 2, 3],
        )


def test_make_multisector_stoks_lt_makers():
    """Test make_multisector stocks less than makers."""
    with pytest.raises(ValueError):
        datasets.make_multisector(
            datasets.RissoUniform(random_state=42),
            datasets.RissoNormal(random_state=42),
            stocks=1,
        )
