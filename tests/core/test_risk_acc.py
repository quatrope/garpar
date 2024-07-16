# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.core import div_acc

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# DRISK TESTS
# =============================================================================


def test_RiskAccessor_stock_beta(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=3)
    expected = pd.Series(
        [0.42396259758517374, 1.748117962092154, 0.8279194403226721],
        name="beta",
        index=["S0", "S1", "S2"],
    )
    expected.index.name = "Stocks"
    pd.testing.assert_series_equal(pf.risk.stock_beta(), expected)


def test_RiskAccessor_stock_beta_force_another_market_column(
    risso_portfolio,
):
    pf = risso_portfolio(random_state=42, stocks=4)
    other_pf = pf.copy(stocks=["_mkt_", "_mkt_0_", "_mkt_1_", "_mkt_2_"])
    np.testing.assert_allclose(other_pf.risk.pf_beta(), pf.risk.pf_beta())


def test_RiskAccessor_portfolio_beta(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 1.0
    np.testing.assert_allclose(pf.risk.portfolio_beta(), expected)


def test_RiskAccessor_treynor_ratio(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = -0.51879158
    np.testing.assert_allclose(pf.risk.treynor_ratio(), expected)


def test_RiskAccessor_portfolio_variance(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 0.01078666
    np.testing.assert_allclose(pf.risk.portfolio_variance(), expected)


def test_RiskAccessor_sharpe_ratio(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = -4.802591
    np.testing.assert_allclose(pf.risk.sharpe_ratio(), expected)


def test_RiskAccessor_value_at_risk(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=3)
    expected = pd.Series(
        [0.0016324583798860148, 0.0030751474549569613, 0.0037714120615394142],
        name="VaR",
        index=["S0", "S1", "S2"],
    )
    expected.index.name = "Stocks"

    pd.testing.assert_series_equal(pf.risk.value_at_risk(), expected)
