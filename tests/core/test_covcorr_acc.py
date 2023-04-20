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


def test_CovarianceAccessor_sample_cov(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.000418, -0.000034],
            [-0.000034, 0.002269],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.cov.sample_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_exp_cov(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [3.32771520e-04, -1.72111801e-05],
            [-1.72111801e-05, 1.82601048e-03],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.cov.exp_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_semi_cov(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [2.74891470e-04, 5.16621976e-05],
            [5.16621976e-05, 6.18762348e-04],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.cov.semi_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_ledoit_wolf_cov(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [9.57760401e-04, -4.25156999e-06],
            [-4.25156999e-06, 1.19172118e-03],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.cov.ledoit_wolf_cov(), expected, rtol=1e-05, atol=1e-06
    )


def test_CovarianceAccessor_oracle_approximating_cov(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.00107474, -0.0],
            [-0.0, 0.00107474],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.cov.oracle_approximating_cov(), expected, rtol=1e-05, atol=1e-06
    )


# =============================================================================
# CORRELATION TESTS
# =============================================================================


def test_CorrelationAccessor_sample_corr(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.9999999999999999, -0.034569766559457114],
            [-0.034569766559457114, 1.0000000000000002],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.corr.sample_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_exp_corr(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.9999999999999999, -0.022079332615257883],
            [-0.022079332615257883, 1.0000000000000002],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.corr.exp_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_semi_corr(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [1.0, 0.1252651622343405],
            [0.1252651622343405, 1.0000000000000002],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.corr.semi_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_ledoit_wolf_corr(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.9999999999999998, -0.003979546336128718],
            [-0.003979546336128719, 0.9999999999999998],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.corr.ledoit_wolf_corr(), expected, rtol=1e-05, atol=1e-06
    )


def test_CorrelationAccessor_oracle_approximating_corr(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    expected = pd.DataFrame(
        [
            [0.9999999999999998, 0.0],
            [0.0, 0.9999999999999998],
        ],
        index=["S0", "S1"],
        columns=["S0", "S1"],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Stocks"

    pd.testing.assert_frame_equal(
        pf.corr.oracle_approximating_corr(), expected, rtol=1e-05, atol=1e-06
    )
