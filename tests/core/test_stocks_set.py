# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test StocksSet module."""

# =============================================================================
# IMPORTS
# =============================================================================

from io import BytesIO

from garpar.core.stocks_set import StocksSet
from garpar.garpar_io import read_hdf5

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_StocksSet_creation():
    """Test StocksSet creation."""
    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})
    weights = [1]
    entropy = [0.5]
    window_size = None
    metadata = {"foo": "faa"}

    manual_ss = StocksSet(
        prices_df=df.copy(),
        weights=weights,
        entropy=entropy,
        window_size=window_size,
        metadata=metadata,
    )
    mk_ss = StocksSet.from_dfkws(
        prices=df.copy(), weights=1, entropy=0.5, window_size=None, **metadata
    )

    assert manual_ss == mk_ss
    assert repr(mk_ss.metadata) == "<metadata {'foo'}>"


def test_StocksSet_len():
    """Test StocksSet len."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    assert len(ss) == 5


def test_StocksSet_copy_eq_ne():
    """Test StocksSet copy eq not equal."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )
    copy = ss.copy()

    assert ss == copy
    assert ss is not copy

    other = StocksSet.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.25,
        window_size=5,
    )

    assert ss != other


def test_StocksSet_bad_weights():
    """Test StocksSet bad weights."""
    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError):
        StocksSet.from_dfkws(df, weights=[1, 2, 3])


def test_StocksSet_slice():
    """Test StocksSet slice."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5], "stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    expected = StocksSet.from_dfkws(
        prices=pd.DataFrame(
            {"stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    result = ss["stock1"]
    assert expected == result


def test_StocksSet_as_returns():
    """Test StocksSet as_returns function."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
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

    result = ss.as_returns()
    pd.testing.assert_frame_equal(result, expected)


def test_StocksSet_as_prices():
    """Test StocksSet as_prices."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
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

    result = ss.as_prices()
    pd.testing.assert_frame_equal(result, expected)


def test_StocksSet_repr():
    """Test StocksSet repr."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    expected = (
        "Stocks  stock[W 1.0, H 0.5]\n"
        "Days                       \n"
        "0                         1\n"
        "1                         2\n"
        "2                         3\n"
        "3                         4\n"
        "4                         5\n"
        "StocksSet [5 days x 1 stocks - W.Size 5]"
    )

    result = repr(ss)

    assert result == expected


def test_StocksSet_to_dataframe():
    """Test StocksSet to_dataframe."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5], "stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
        foo="zaraza",
    )

    expected = pd.DataFrame(
        {
            "stock0": [1, 0.5, 5, 1, 2, 3, 4, 5],
            "stock1": [1, 0.5, 5, 10, 20, 30, 40, 50],
        },
        index=["Weights", "Entropy", "WSize", 0, 1, 2, 3, 4],
    )
    expected_attrs = {"__garpar_metadata__": {"foo": "zaraza"}}

    result = ss.to_dataframe()
    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs == expected_attrs


def test_StocksSet_to_hdf5():
    """Test StocksSet to_hdf5."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5], "stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    buff = BytesIO()
    ss.to_hdf5(buff)
    buff.seek(0)
    result = read_hdf5(buff)

    assert ss == result


def test_StocksSet_weights_prune():
    """Test StocksSet weights_prune."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
            {
                "stock0": [1, 2, 3, 4, 5],
                "stock1": [10, 20, 30, 40, 50],
                "stock2": [10, 20, 30, 40, 50],
            },
        ),
        weights=[0.7, 0.29999, 0.000001],
        entropy=0.5,
        window_size=5,
    )

    pss = ss.wprune()
    np.testing.assert_array_equal(pss.stocks, ["stock0", "stock1"])


def test_StocksSet_delisted_prune():
    """Test StocksSet delisted_prune."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
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

    pss = ss.dprune()

    np.testing.assert_array_equal(pss.stocks, ["stock0", "stock2"])


def test_StocksSet_scale_weights():
    """Test StocksSet scale_weights."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
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

    swss = ss.scale_weights()

    np.testing.assert_array_equal(swss.weights, [0.1, 0.75, 0.15])
    np.testing.assert_array_equal(swss.weights.sum(), 1.0)


def test_StocksSet_scale_weights_bad_scaler():
    """Test StocksSet scale_weights with bad scaler."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
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

    with pytest.raises(ValueError):
        ss.scale_weights(scaler=None)


def test_StocksSet_refresh_entropy():
    """Test StocksSet refresh_entropy."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
            {
                "stock0": [1, 2, 3, 4, 5],
                "stock1": [10, 20, 30, 40, 50],
                "stock2": [10, 20, 30, 40, 50],
            },
        ),
        weights=[10, 75, 15],
        entropy=0.5,
        window_size=None,
    )

    swss = ss.refresh_entropy()
    np.testing.assert_allclose(
        swss.entropy.values, [1.48975, 1.48975, 1.48975], atol=1e-6
    )


def test_StocksSet_refresh_entropy_bad_entropy():
    """Test StocksSet refresh_entropy with bad entropy."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame(
            {
                "stock0": [1, 2, 3, 4, 5],
                "stock1": [10, 20, 30, 40, 50],
                "stock2": [10, 20, 30, 40, 50],
            },
        ),
        weights=[10, 75, 15],
        entropy=0.5,
        window_size=None,
    )

    with pytest.raises(ValueError):
        ss.refresh_entropy(entropy=None)


def test_StocksSet_repr_html():
    """Test StocksSet repr_html."""
    ss = StocksSet.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
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
        "      <th>stock[W 1.0, H 0.5]</th>\n"
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
        "</div><em class='portfolio-dim'>5 days x 1 stocks - W.Size 5</em>\n"
        "</div>"
    )

    result = ss._repr_html_()

    assert result == expected
