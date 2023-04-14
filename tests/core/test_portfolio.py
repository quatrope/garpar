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
from garpar.core.portfolio import GARPAR_METADATA_KEY, Bunch, Portfolio

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_Portfolio_creation():

    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})
    df.attrs[GARPAR_METADATA_KEY] = Bunch(
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
    assert repr(mk_pf.metadata) == "metadata(entropy, window_size)"


def test_Portfolio_len():

    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    assert len(pf) == 5


def test_Portfolio_dir():

    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    assert not set(["entropy", "window_size"]).difference(dir(pf.metadata))


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
        Portfolio(df, [1])

    with pytest.raises(ValueError):
        Portfolio.from_dfkws(df, [1], foo="ggg")

    with pytest.raises(TypeError):
        Portfolio.from_dfkws(df, [1], entropy="ggg")


def test_Portfolio_bad_weights():
    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})
    df.attrs[GARPAR_METADATA_KEY] = {}

    with pytest.raises(ValueError):
        Portfolio(df, weights=[1, 2, 3])


def test_Portfolio_slice():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5], "stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    expected = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {"stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    result = pf["stock1"]
    assert expected == result


def test_Portfolio_as_returns():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5]},
        ),
        entropy=0.5,
        window_size=5,
    )

    expected = pd.DataFrame(
        {"stock0": [1.000000, 0.500000, 0.333333, 0.250000]},
        index=[1, 2, 3, 4],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Days"

    result = pf.as_returns()
    pd.testing.assert_frame_equal(result, expected)


def test_Portfolio_as_prices():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5]},
        ),
        entropy=0.5,
        window_size=5,
    )

    expected = pd.DataFrame(
        {"stock0": [1, 2, 3, 4, 5]},
        index=[0, 1, 2, 3, 4],
    )
    expected.columns.name = "Stocks"
    expected.index.name = "Days"

    result = pf.as_prices()
    pd.testing.assert_frame_equal(result, expected)


def test_Portfolio_repr():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    expected = (
        "Stocks  stock[\u2696 1.0]\n"
        "Days                \n"
        "0                  1\n"
        "1                  2\n"
        "2                  3\n"
        "3                  4\n"
        "4                  5\n"
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


def test_Portfolio_wprune():
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

    ppf = pf.wprune()

    assert np.all(ppf.stocks == ["stock0", "stock1"])


def test_Portfolio_dprune():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1, 2, 3, 4, 5],
                "stock1": [10, 20, 30, 0, 0],
                "stock2": [10, 20, 30, 40, 50],
            },
        ),
        weights=[0.7, 0.29999, 0.00001],
        entropy=0.5,
        window_size=5,
    )

    ppf = pf.dprune()

    assert np.all(ppf.stocks == ["stock0", "stock2"])


def test_Portfolio_scale_weights():

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
    assert np.isclose(swpf.weights.sum(), 1.0)


def test_Portfolio_repr_html():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    expected = (
        "<div class='portfolio'>\n"
        "<div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th>Stocks</th>\n"
        "      <th>stock[⚖ 1.0]</th>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>Days</th>\n"
        "      <th></th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>1</th>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>2</th>\n"
        "      <td>3</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>3</th>\n"
        "      <td>4</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>4</th>\n"
        "      <td>5</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div><em class='portfolio-dim'>5 days x 1 stocks</em>\n"
        "</div>"
    )

    result = pf._repr_html_()

    assert result == expected
