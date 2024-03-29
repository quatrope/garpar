# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


def test_ExpectedReturnsAccessor_capm(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.Series(
        [0.01624699889633035, -0.0018563108280785154],
        name="CAPM",
        index=["S0", "S1"],
    )
    expected.index.name = "Stocks"

    result = pf.ereturns.capm()

    pd.testing.assert_series_equal(result, expected)


def test_ExpectedReturnsAccessor_mah(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.Series(
        [-0.0562953805511921, 0.07434810133853742],
        name="MAH",
        index=["S0", "S1"],
    )
    expected.index.name = "Stocks"

    result = pf.ereturns.mah()

    pd.testing.assert_series_equal(result, expected)


def test_ExpectedReturnsAccessor_emah(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.Series(
        [-0.05589539026726009, 0.07853139028034328],
        name="EMAH",
        index=["S0", "S1"],
    )
    expected.index.name = "Stocks"

    result = pf.ereturns.emah()

    pd.testing.assert_series_equal(result, expected)
