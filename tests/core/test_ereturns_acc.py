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


# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd

# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


def test_ExpectedReturnsAccessor_capm(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.Series(
        [0.069596, -0.468277],
        name="CAPM",
        index=["S0", "S1"],
    )
    expected.index.name = "Stocks"

    result = ss.ereturns.capm()

    pd.testing.assert_series_equal(result, expected)


def test_ExpectedReturnsAccessor_mah(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.Series(
        [-0.726869, 1.292289],
        name="MAH",
        index=["S0", "S1"],
    )
    expected.index.name = "Stocks"

    result = ss.ereturns.mah()

    pd.testing.assert_series_equal(result, expected)


def test_ExpectedReturnsAccessor_emah(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.Series(
        [-0.724145, 1.326126],
        name="EMAH",
        index=["S0", "S1"],
    )
    expected.index.name = "Stocks"

    result = ss.ereturns.emah()

    pd.testing.assert_series_equal(result, expected)
