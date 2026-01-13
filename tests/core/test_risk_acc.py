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

"""Test RiskAccessor module."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# DRISK TESTS
# =============================================================================


def test_RiskAccessor_stock_beta(risso_stocks_set):
    """Test RiskAccessor stock beta."""
    ss = risso_stocks_set(random_state=42, stocks=3)
    expected = pd.Series(
        [1.901328, 0.622217, 0.476455],
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
    expected = 1.19352366732936
    np.testing.assert_allclose(ss.risk.stocks_set_beta(), expected)


@pytest.fixture(scope="function")
def test_RiskAccessor_treynor_ratio(risso_stocks_set):
    """Test RiskAccessor treynor ratio."""
    ss = risso_stocks_set(random_state=42)
    expected = -0.40175717770374153
    np.testing.assert_allclose(ss.risk.treynor_ratio(), expected)


def test_RiskAccessor_portfolio_variance(risso_stocks_set):
    """Test RiskAccessor portfolio variance."""
    ss = risso_stocks_set(random_state=42)
    expected = 0.014769759602379556
    np.testing.assert_allclose(ss.risk.stocks_set_variance(), expected)


@pytest.fixture(scope="function")
def test_RiskAccessor_sharpe_ratio(risso_stocks_set):
    """Test RiskAccessor sharpe ratio."""
    ss = risso_stocks_set(random_state=42)
    expected = -3.9455537724146836
    np.testing.assert_allclose(ss.risk.sharpe_ratio(), expected)


def test_RiskAccessor_value_at_risk(risso_stocks_set):
    """Test RiskAccessor value at risk."""
    ss = risso_stocks_set(random_state=42, stocks=3)
    expected = pd.Series(
        [0.01111180879317486, 0.013124965253070608, 0.009917326684708794],
        name="VaR",
        index=["S0", "S1", "S2"],
    )
    expected.index.name = "Stocks"

    pd.testing.assert_series_equal(ss.risk.value_at_risk(), expected)
