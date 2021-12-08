# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import attr

from garpar.portfolio import GARPAR_METADATA_KEY, Metadata, Portfolio

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_Portfilio_creation():

    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})
    df.attrs[GARPAR_METADATA_KEY] = Metadata(
        initial_prices=pd.Series({"stock": [1]}), entropy=0.5, window_size=5
    )

    manual_pf = Portfolio(df=df.copy())

    mk_pf = Portfolio.from_dfkws(
        df=df,
        initial_prices=pd.Series({"stock": [1]}),
        entropy=0.5,
        window_size=5,
    )

    assert manual_pf == mk_pf


def test_Portfolio_copy_eq_ne():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        initial_prices=pd.Series({"stock": [1]}),
        entropy=0.5,
        window_size=5,
    )
    copy = pf.copy()

    assert (
        pf == copy
        and pf is not copy
        and pf._df.attrs[GARPAR_METADATA_KEY]
        == copy._df.attrs[GARPAR_METADATA_KEY]
        and pf._df.attrs[GARPAR_METADATA_KEY]
        is not copy._df.attrs[GARPAR_METADATA_KEY]
    )

    other = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        initial_prices=pd.Series({"stock": [2]}),
        entropy=0.5,
        window_size=5,
    )

    assert pf != other


def test_Portfolio_bad_metadata():
    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})
    df.attrs[GARPAR_METADATA_KEY] = None

    with pytest.raises(TypeError):
        Portfolio(df)


def test_Portfolio_access_df():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        initial_prices=pd.Series({"stock": [1]}),
        entropy=0.5,
        window_size=5,
    )

    pd.testing.assert_frame_equal(pf.describe(), pf._df.describe())


def test_Portfolio_dir():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        initial_prices=pd.Series({"stock": [1]}),
        entropy=0.5,
        window_size=5,
    )

    pf_dir = dir(pf)
    df_dir = dir(pf._df)
    meta_dir = attr.asdict(pf._df.attrs[GARPAR_METADATA_KEY])

    assert set(pf_dir).issuperset(df_dir)
    assert set(pf_dir).issuperset(meta_dir)


def test_Portfolio_repr():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        initial_prices=pd.Series({"stock": [1]}),
        entropy=0.5,
        window_size=5,
    )

    expected = (
        "   stock\n"
        "0      1\n"
        "1      2\n"
        "2      3\n"
        "3      4\n"
        "4      5\n"
        "Portfolio [5 days x 1 stocks]"
    )

    result = repr(pf)
    assert result == expected
