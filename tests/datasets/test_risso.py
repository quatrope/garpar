# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================


from garpar import datasets, Portfolio

import numpy as np

import pandas as pd


# =============================================================================
# TESTS
# =============================================================================


def test_argnearest():
    assert datasets.risso.argnearest([0.1, 0.11, -0.001], 0) == 2


# =============================================================================
# UNIFORM
# =============================================================================


def test_RissoUniform_delisted():
    maker = datasets.RissoUniform()
    assert maker.make_stock_price(0, True, np.random.default_rng()) == 0


def test_make_risso_uniform():
    result = datasets.make_risso_uniform(
        random_state=42, window_size=2, days=3, stocks=2
    )
    expected = Portfolio.from_dfkws(
        pd.DataFrame(
            [
                [100.000000, 100.000000],
                [103.992997, 95.595250],
                [99.728335, 92.098338],
                [102.602648, 93.361968],
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
    maker = datasets.RissoNormal()
    assert maker.make_stock_price(0, True, np.random.default_rng()) == 0


def test_make_risso_normal():
    result = datasets.make_risso_normal(
        random_state=42, window_size=2, days=3, stocks=2
    )
    expected = Portfolio.from_dfkws(
        pd.DataFrame(
            [
                [100.0, 100.0],
                [100.15141604581204, 99.73470995577729],
                [100.04597610435215, 99.52898003797632],
                [100.19705644445486, 99.83649478347202],
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
    maker = datasets.RissoLevyStable()
    assert maker.make_stock_price(0, True, np.random.default_rng()) == 0


def test_make_risso_levy_stable():
    result = datasets.make_risso_levy_stable(
        random_state=42, window_size=2, days=3, stocks=2
    )
    expected = Portfolio.from_dfkws(
        pd.DataFrame(
            [
                [100.0, 100.0],
                [100.00553617562464, 99.98702371323826],
                [100.00527303047784, 99.97349761712346],
                [100.01232581754685, 99.98159967977507],
            ],
            columns=["S0", "S1"],
        ),
        weights=[1, 1],
    )

    pd.testing.assert_frame_equal(result.as_prices(), expected.as_prices(), check_exact=False, atol=.01, rtol=.01)
    pd.testing.assert_series_equal(result.weights, expected.weights)
