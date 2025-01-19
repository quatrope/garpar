# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test RiskAccessor module."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

# =============================================================================
# DRISK TESTS
# =============================================================================


def test_RiskAccessor_stock_beta(risso_stocks_set):
    """Test RiskAccessor stock beta."""
    ss = risso_stocks_set(random_state=42, stocks=3)
    expected = pd.Series(
        [1.494153, 1.398759, 0.107088],
        name="beta",
        index=["S0", "S1", "S2"],
    )
    expected.index.name = "Stocks"
    pd.testing.assert_series_equal(ss.risk.stock_beta(), expected)


def test_RiskAccessor_stock_beta_force_another_market_column(
    risso_stocks_set,
):
    """Test RiskAccessor stock beta force another market column."""
    ss = risso_stocks_set(random_state=42, stocks=4)
    other_ss = ss.copy(stocks=["_mkt_", "_mkt_0_", "_mkt_1_", "_mkt_2_"])
    np.testing.assert_allclose(other_ss.risk.ss_beta(), ss.risk.ss_beta())


def test_RiskAccessor_portfolio_beta(risso_stocks_set):
    """Test RiskAccessor portfolio beta."""
    ss = risso_stocks_set(random_state=42)
    expected = 2.1417061348856747
    np.testing.assert_allclose(ss.risk.stocks_set_beta(), expected)


def test_RiskAccessor_treynor_ratio(risso_stocks_set):
    """Test RiskAccessor treynor ratio."""
    ss = risso_stocks_set(random_state=42)
    expected = 0.07097715626252858
    np.testing.assert_allclose(ss.risk.treynor_ratio(), expected)


def test_RiskAccessor_portfolio_variance(risso_stocks_set):
    """Test RiskAccessor portfolio variance."""
    ss = risso_stocks_set(random_state=42)
    expected = 0.0016910095064705654
    np.testing.assert_allclose(ss.risk.stocks_set_variance(), expected)


def test_RiskAccessor_sharpe_ratio(risso_stocks_set):
    """Test RiskAccessor sharpe ratio."""
    ss = risso_stocks_set(random_state=42)
    expected = 4.182984484028877
    np.testing.assert_allclose(ss.risk.sharpe_ratio(), expected)


def test_RiskAccessor_value_at_risk(risso_stocks_set):
    """Test RiskAccessor value at risk."""
    ss = risso_stocks_set(random_state=42, stocks=3)
    expected = pd.Series(
        [0.0011073872112337124, 0.0031327467496115036, 0.0022920636466858824],
        name="VaR",
        index=["S0", "S1", "S2"],
    )
    expected.index.name = "Stocks"

    pd.testing.assert_series_equal(ss.risk.value_at_risk(), expected)
