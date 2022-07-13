# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from numpy import exp
from garpar.optimize import BlackLitterman, Markowitz, OptimizerABC
from garpar.core import Portfolio

import pandas as pd
import pandas.testing as pdt

import pytest

# =============================================================================
# TESTS OPTIMIZER
# =============================================================================


def test_OptimizerABC_not_implementhed_methods():
    class Foo(OptimizerABC):
        def serialize(self, port):
            return super().serialize(port)

        def deserialize(self, port, weights):
            return super().deserialize(port, weights)

        def optimize(self, port):
            return super().optimize(port)

    opt = Foo()
    with pytest.raises(NotImplementedError):
        opt.serialize(0)

    with pytest.raises(NotImplementedError):
        opt.optimize(0)

    with pytest.raises(NotImplementedError):
        opt.deserialize(0, 0)


# =============================================================================
# TESTS MARKOWITZ
# =============================================================================


def test_Markowitz_is_OptimizerABC():
    assert issubclass(Markowitz, OptimizerABC)


def test_Markowitz_defaults():
    markowitz = Markowitz()

    assert markowitz.weight_bounds == (0, 1)
    assert markowitz.market_neutral is False


def test_Markowitz_serialize():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            },
        ),
        entropy=0.5,
        window_size=5,
    )

    # Instance
    markowitz = Markowitz()

    # Tested method
    result = markowitz.serialize(pf)

    # Expectations
    expected_mu = pd.Series({"stock0": 46.121466, "stock1": 287.122362})
    expected_mu.name = markowitz.returns.upper()
    expected_mu.index.name = "Stocks"

    expected_cov = pd.DataFrame(
        data={
            "stock0": [0.17805911, -0.13778805],
            "stock1": [-0.13778805, 0.13090794],
        },
        index=["stock0", "stock1"],
    )
    expected_cov.name = markowitz.covariance.upper()
    expected_cov.index.name = "Stocks"
    expected_cov.columns.name = "Stocks"



    # Assert
    assert isinstance(result, dict)
    assert result.keys() == {"expected_returns", "cov_matrix", "weight_bounds"}
    pdt.assert_series_equal(expected_mu, result["expected_returns"])
    pdt.assert_frame_equal(expected_cov, result["cov_matrix"])
    assert result["weight_bounds"] == (0, 1)


def test_Markowitz_deserialize():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            },
        ),
        weights=[0.5, 0.5],
        entropy=0.5,
        window_size=5,
    )

    # Instance
    markowitz = Markowitz()

    # Tested method
    weights = {"stock0": 0.45966836, "stock1": 0.54033164}
    result = markowitz.deserialize(pf, weights)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45966836, "stock1": 0.54033164}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not pf
    assert isinstance(result, Portfolio)
    pdt.assert_frame_equal(pf._df, result._df)

    assert isinstance(result.weights, pd.Series)
    pdt.assert_series_equal(result.weights, expected_weights)


def test_Markowitz_optimize():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            },
        ),
        weights=[0.5, 0.5],
        entropy=0.5,
        window_size=5,
    )

    # Instance
    markowitz = Markowitz()

    # Tested method
    result = markowitz.optimize(pf, target_return=1.0)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45966836, "stock1": 0.54033164}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not pf
    assert isinstance(result, Portfolio)
    pdt.assert_frame_equal(pf._df, result._df)

    assert isinstance(result.weights, pd.Series)
    pdt.assert_series_equal(result.weights, expected_weights)


# =============================================================================
# TESTS BLACK LITTERMAN
# =============================================================================


def test_BlackLitterman_is_OptimizerABC():
    assert issubclass(BlackLitterman, OptimizerABC)


def test_BlackLitterman_defaults():
    bl = BlackLitterman()

    assert bl.prior == "equal"
    assert bl.absolute_views is None
    assert bl.P is None
    assert bl.Q is None


def test_BlackLitterman_serialize():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            },
        ),
        entropy=0.5,
        window_size=5,
    )

    # Instance
    viewdict = {"stock0": 0.01, "stock1": 0.03}
    bl = BlackLitterman(prior="algo", absolute_views=viewdict)

    # Tested method
    result = bl.serialize(pf)

    # Expectations
    expected_views = {"stock0": 0.01, "stock1": 0.03}
    expected_cov = pd.DataFrame(
        data={
            "stock0": [0.17805911, -0.13778805],
            "stock1": [-0.13778805, 0.13090794],
        },
        index=["stock0", "stock1"],
    )
    expected_cov.index.name = "Stocks"
    expected_cov.columns.name = "Stocks"

    # Assert
    assert isinstance(result, dict)
    assert result.keys() == {"pi", "absolute_views", "cov_matrix", "P", "Q"}

    assert result["pi"] == "algo"
    assert result["absolute_views"] == expected_views
    pdt.assert_frame_equal(expected_cov, result["cov_matrix"])
    assert result["P"] is None
    assert result["Q"] is None


def test_BlackLitterman_deserialize():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            },
        ),
        entropy=0.5,
        window_size=5,
    )

    # Instance
    viewdict = {"stock0": 0.01, "stock1": 0.03}
    prior = pd.Series(data={"stock0": 0.02, "stock1": 0.04})
    bl = BlackLitterman(prior=prior, absolute_views=viewdict)

    # Tested method
    weights = {"stock0": 0.45157882, "stock1": 0.54842117}
    result = bl.deserialize(pf, weights)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45157882, "stock1": 0.54842117}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not pf
    assert isinstance(result, Portfolio)
    pdt.assert_frame_equal(pf._df, result._df)

    assert isinstance(result.weights, pd.Series)
    pdt.assert_series_equal(result.weights, expected_weights)


def test_BlackLitterman_optimize():
    pf = Portfolio.from_dfkws(
        df=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            },
        ),
        entropy=0.5,
        window_size=5,
    )

    # Instance
    viewdict = {"stock0": 0.01, "stock1": 0.03}
    prior = pd.Series(data={"stock0": 0.02, "stock1": 0.04})
    bl = BlackLitterman(prior=prior, absolute_views=viewdict)

    # Tested method
    result = bl.optimize(pf)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45157882, "stock1": 0.54842117}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not pf
    assert isinstance(result, Portfolio)
    pdt.assert_frame_equal(pf._df, result._df)

    assert isinstance(result.weights, pd.Series)
    pdt.assert_series_equal(result.weights, expected_weights)
