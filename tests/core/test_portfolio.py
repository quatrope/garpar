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
    weights = [1]
    entropy = [0.5]
    window_size = None
    metadata = {"foo": "faa"}

    manual_pf = Portfolio(
        prices_df=df.copy(),
        weights=weights,
        entropy=entropy,
        window_size=window_size,
        metadata=metadata,
    )
    mk_pf = Portfolio.from_dfkws(
        prices=df.copy(), weights=1, entropy=0.5, window_size=None, **metadata
    )

    assert manual_pf == mk_pf
    assert repr(mk_pf.metadata) == "<metadata {'foo'}>"


def test_Portfolio_len():
    pf = Portfolio.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )

    assert len(pf) == 5


def test_Portfolio_copy_eq_ne():
    pf = Portfolio.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.5,
        window_size=5,
    )
    copy = pf.copy()

    assert pf == copy
    assert pf is not copy

    other = Portfolio.from_dfkws(
        prices=pd.DataFrame({"stock": [1, 2, 3, 4, 5]}),
        entropy=0.25,
        window_size=5,
    )

    assert pf != other


def test_Portfolio_bad_weights():
    df = pd.DataFrame({"stock": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError):
        Portfolio.from_dfkws(df, weights=[1, 2, 3])


def test_Portfolio_slice():
    pf = Portfolio.from_dfkws(
        prices=pd.DataFrame(
            {"stock0": [1, 2, 3, 4, 5], "stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    expected = Portfolio.from_dfkws(
        prices=pd.DataFrame(
            {"stock1": [10, 20, 30, 40, 50]},
        ),
        entropy=0.5,
        window_size=5,
    )

    result = pf["stock1"]
    assert expected == result


def test_Portfolio_as_returns():
    pf = Portfolio.from_dfkws(
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

    result = pf.as_returns()
    pd.testing.assert_frame_equal(result, expected)


def test_Portfolio_as_prices():
    pf = Portfolio.from_dfkws(
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

    result = pf.as_prices()
    pd.testing.assert_frame_equal(result, expected)


def test_Portfolio_repr():
    pf = Portfolio.from_dfkws(
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
        "Portfolio [5 days x 1 stocks - W.Size 5]"
    )

    result = repr(pf)

    assert result == expected


def test_Portfolio_to_dataframe():
    pf = Portfolio.from_dfkws(
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

    result = pf.to_dataframe()
    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs == expected_attrs


def test_Portfolio_to_hdf5():
    pf = Portfolio.from_dfkws(
        prices=pd.DataFrame(
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


def test_Portfolio_weights_prune():
    pf = Portfolio.from_dfkws(
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

    ppf = pf.wprune()
    np.testing.assert_array_equal(ppf.stocks, ["stock0", "stock1"])


def test_Portfolio_delisted_prune():
    pf = Portfolio.from_dfkws(
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

    ppf = pf.dprune()

    np.testing.assert_array_equal(ppf.stocks, ["stock0", "stock2"])


def test_Portfolio_scale_weights():
    pf = Portfolio.from_dfkws(
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

    swpf = pf.scale_weights()

    np.testing.assert_array_equal(swpf.weights, [0.1, 0.75, 0.15])
    np.testing.assert_array_equal(swpf.weights.sum(), 1.0)


def test_Portfolio_scale_weights_bad_scaler():
    pf = Portfolio.from_dfkws(
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
        pf.scale_weights(scaler=None)


def test_Portfolio_refresh_entropy():
    pf = Portfolio.from_dfkws(
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

    swpf = pf.refresh_entropy()
    np.testing.assert_allclose(
        swpf.entropy.values, [1.48975, 1.48975, 1.48975], atol=1e-6
    )


def test_Portfolio_refresh_entropy_bad_entropy():
    pf = Portfolio.from_dfkws(
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
        pf.refresh_entropy(entropy=None)


def test_Portfolio_repr_html():
    pf = Portfolio.from_dfkws(
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

    result = pf._repr_html_()

    assert result == expected
