# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from inspect import Attribute
from garpar.core import prices_acc

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


@pytest.mark.parametrize("metric", prices_acc.PricesAccessor._DF_WHITELIST)
def test_PricesAccessor_df_whitelist(risso_portfolio, metric):
    pf = risso_portfolio(random_state=42, stocks=2)
    result_call = pf.prices(metric)
    result_getattr = getattr(pf.prices, metric)()
    result_df = getattr(pf._prices_df, metric)()

    if isinstance(result_call, pd.DataFrame):
        pd.testing.assert_frame_equal(result_call, result_getattr)
        pd.testing.assert_frame_equal(result_call, result_df)
    elif isinstance(result_call, pd.Series):
        pd.testing.assert_series_equal(result_call, result_getattr)
        pd.testing.assert_series_equal(result_call, result_df)
    else:
        assert result_call == result_getattr == result_df


def test_PricesAccessor_invalid_metric(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)
    with pytest.raises(AttributeError):
        pf.prices.getattr.zaraza()


def test_PricesAccessor_log(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    result_call = pf.prices("log")
    result_getattr = pf.prices.log()
    result_df = pf._prices_df.apply(np.log)

    pd.testing.assert_frame_equal(result_call, result_getattr)
    pd.testing.assert_frame_equal(result_call, result_df)


def test_PricesAccessor_log10(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    result_call = pf.prices("log10")
    result_getattr = pf.prices.log10()
    result_df = pf._prices_df.apply(np.log10)

    pd.testing.assert_frame_equal(result_call, result_getattr)
    pd.testing.assert_frame_equal(result_call, result_df)


def test_PricesAccessor_log2(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    result_call = pf.prices("log2")
    result_getattr = pf.prices.log2()
    result_df = pf._prices_df.apply(np.log2)

    pd.testing.assert_frame_equal(result_call, result_getattr)
    pd.testing.assert_frame_equal(result_call, result_df)


def test_PricesAccessor_log2(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    result_call = pf.prices("log2")
    result_getattr = pf.prices.log2()
    result_df = pf._prices_df.apply(np.log2)

    pd.testing.assert_frame_equal(result_call, result_getattr)
    pd.testing.assert_frame_equal(result_call, result_df)


def test_PricesAccessor_mad(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    result_call = pf.prices("mad")
    result_getattr = pf.prices.mad()

    pd.testing.assert_series_equal(result_call, result_getattr)


def test_PricesAccessor_dir(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    white_list = set(prices_acc.PricesAccessor._DF_WHITELIST)
    dir_pf_prices = dir(pf.prices)

    assert not white_list.difference(dir_pf_prices)
