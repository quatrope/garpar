# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.core import prices_acc

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


@pytest.mark.parametrize("metric", prices_acc.PricesAccessor._DF_WHITELIST)
def test_PricesAccessor_df_whitelist(risso_stocks_set, metric):
    ss = risso_stocks_set(random_state=42, stocks=2)
    result_call = ss.prices(metric)
    result_getattr = getattr(ss.prices, metric)()
    result_df = getattr(ss._prices_df, metric)()

    if isinstance(result_call, pd.DataFrame):
        pd.testing.assert_frame_equal(result_call, result_getattr)
        pd.testing.assert_frame_equal(result_call, result_df)
    elif isinstance(result_call, pd.Series):
        pd.testing.assert_series_equal(result_call, result_getattr)
        pd.testing.assert_series_equal(result_call, result_df)
    else:
        assert result_call == result_getattr == result_df


def test_PricesAccessor_invalid_metric(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)
    with pytest.raises(AttributeError):
        ss.prices.getattr.zaraza()


def test_PricesAccessor_log(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    result_call = ss.prices("log")
    result_getattr = ss.prices.log()
    result_df = ss._prices_df.apply(np.log)

    pd.testing.assert_frame_equal(result_call, result_getattr)
    pd.testing.assert_frame_equal(result_call, result_df)


def test_PricesAccessor_log10(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    result_call = ss.prices("log10")
    result_getattr = ss.prices.log10()
    result_df = ss._prices_df.apply(np.log10)

    pd.testing.assert_frame_equal(result_call, result_getattr)
    pd.testing.assert_frame_equal(result_call, result_df)


def test_PricesAccessor_log2(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    result_call = ss.prices("log2")
    result_getattr = ss.prices.log2()
    result_df = ss._prices_df.apply(np.log2)

    pd.testing.assert_frame_equal(result_call, result_getattr)
    pd.testing.assert_frame_equal(result_call, result_df)


def test_PricesAccessor_mad(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    result_call = ss.prices("mad")
    result_getattr = ss.prices.mad()

    pd.testing.assert_series_equal(result_call, result_getattr)


def test_PricesAccessor_dir(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    white_list = set(prices_acc.PricesAccessor._DF_WHITELIST)
    dir_ss_prices = dir(ss.prices)

    assert not white_list.difference(dir_ss_prices)
