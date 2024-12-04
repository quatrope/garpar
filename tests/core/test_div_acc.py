# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


def test_DiversificationMetricsAccessor_ratio(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = 0.044195384210418555
    np.testing.assert_allclose(ss.div.ratio(), expected)


def test_DiversificationMetricsAccessor_mrc(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=3)

    expected = pd.Series(
        [0.006913, 0.003705, 0.006710],
        name="MRC",
        index=["S0", "S1", "S2"],
    )
    expected.index.name = "Stocks"

    pd.testing.assert_series_equal(
        ss.div.mrc(), expected, rtol=1e-05, atol=1e-06
    )


def test_DiversificationMetricsAccessor_pdi(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = 2.9637573582423076
    np.testing.assert_allclose(ss.div.pdi(), expected)


def test_DiversificationMetricsAccessor_zheng_entropy(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = 2.156003639825938
    np.testing.assert_allclose(ss.div.zheng_entropy(), expected)


def test_DiversificationMetricsAccessor_cross_entropy(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = 0.21781402181785053
    np.testing.assert_allclose(ss.div.cross_entropy(), expected)


def test_DiversificationMetricsAccessor_ke_zang_entropy(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = 2.1576946493324085
    np.testing.assert_allclose(ss.div.ke_zang_entropy(), expected)
