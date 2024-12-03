# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================


import pandas as pd


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


def test_CovarianceAccessor_sample_cov(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.000697, 0.000312],
            [0.000312, 0.000904],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.cov.sample_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_exp_cov(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.000555, 0.000245],
            [0.000245, 0.000723],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.cov.exp_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_semi_cov(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.000813, 0.000214],
            [0.000214, 0.000132],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.cov.semi_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_ledoit_wolf_cov(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.00064, 0.00000],
            [0.00000, 0.00064],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.cov.ledoit_wolf_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_oracle_approximating_cov(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.00064, 0.00000],
            [0.00000, 0.00064],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.cov.oracle_approximating_cov(), expected, rtol=1e-05, atol=1e-06
    )


# =============================================================================
# CORRELATION TESTS
# =============================================================================


def test_CorrelationAccessor_sample_corr(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [1.00000, 0.39243],
            [0.39243, 1.00000],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.corr.sample_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_exp_corr(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [1.000000, 0.387138],
            [0.387138, 1.000000],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.corr.exp_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_semi_corr(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [1.000000, 0.651742],
            [0.651742, 1.000000],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.corr.semi_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_ledoit_wolf_corr(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.corr.ledoit_wolf_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_oracle_approximating_corr(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        ss.corr.oracle_approximating_corr(), expected, rtol=1e-05, atol=1e-06
    )
