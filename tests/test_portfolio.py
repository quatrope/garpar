# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from io import BytesIO

from garpar.io import read_hdf5
from garpar.core.portfolio import GARPAR_METADATA_KEY, Metadata, Portfolio

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_Portfolio_creation():

    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})
    df.attrs[GARPAR_METADATA_KEY] = Metadata(
        {
            "entropy": 0.5,
            "window_size": 5,
        }
    )

    manual_pf = Portfolio(df=df.copy(), weights=[1.0])
    mk_pf = Portfolio.from_dfkws(
        df=df,
        entropy=0.5,
        window_size=5,
    )

    assert manual_pf == mk_pf


def test_Portfolio_copy_eq_ne():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )
    copy = pf.copy()

    assert pf == copy
    assert pf is not copy
    assert (
        pf._df.attrs[GARPAR_METADATA_KEY]
        == copy._df.attrs[GARPAR_METADATA_KEY]
    )
    assert (
        pf._df.attrs[GARPAR_METADATA_KEY]
        is not copy._df.attrs[GARPAR_METADATA_KEY]
    )

    other = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.25,
        window_size=5,
    )

    assert pf != other


def test_Portfolio_bad_metadata():
    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})
    df.attrs[GARPAR_METADATA_KEY] = None

    with pytest.raises(TypeError):
        Portfolio(df)


def test_Portfolio_repr():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    expected = (
        "Stocks stock[\u2696 1.0]\n"
        "Days               \n"
        "0                 1\n"
        "1                 2\n"
        "2                 3\n"
        "3                 4\n"
        "4                 5\n"
        "Portfolio [5 days x 1 stocks]"
    )

    result = repr(pf)
    assert result == expected


def test_Portfolio_to_dataframe():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5], "stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    expected = pd.DataFrame(
        {
            "stock0": [0.5, 0.5, 5, 1, 2, 3, 4, 5],
            "stock1": [0.5, 0.5, 5, 10, 20, 30, 40, 50],
        },
        index=["Weights", "entropy", "window_size", 0, 1, 2, 3, 4],
    )

    result = pf.to_dataframe()
    pd.testing.assert_frame_equal(result, expected)


def test_Portfolio_to_hdf5():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5], "stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    buff = BytesIO()
    pf.to_hdf5(buff)
    buff.seek(0)
    result = read_hdf5(buff)

    assert pf == result


def test_Portfolio_prune():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1, 2, 3, 4, 5],
                "stock1": [10, 20, 30, 40, 50],
                "stock2": [10, 20, 30, 40, 50],
            },
        ),
        weights=[0.7, 0.29999, 0.00001],
        entropy=0.5,
        window_size=5,
    )

    ppf = pf.prune()

    assert np.all(ppf.stocks == ["stock0", "stock1"])


def test_Portfolio_rescaleweights():

    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1, 2, 3, 4, 5],
                "stock1": [10, 20, 30, 40, 50],
                "stock2": [10, 20, 30, 40, 50],
            },
        ),
        weights=[10, 75, 15],
        entropy=0.5,
        window_size=5,
    )

    swpf = pf.scale_weights()

    assert np.all(swpf.weights == [0.1, 0.75, 0.15])
    assert swpf.weights.sum() == 1.0
