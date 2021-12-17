# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.optimize import Markowitz, OptimizerABC
from garpar.optimize import mean_historical_return, sample_covariance
from garpar.portfolio import Portfolio

import pandas as pd
import pandas.testing as pdt

import pytest

# =============================================================================
# TESTS WRAPPER FUNCTIONS
# =============================================================================


def test_mean_historical_return():
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

    result = mean_historical_return(pf)
    expected = pd.Series({"stock0": 46.121466, "stock1": 287.122362})

    pdt.assert_series_equal(result, expected)


def test_sample_covariance():
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

    result = sample_covariance(pf)
    expected = pd.DataFrame(
        data={
            "stock0": [0.17805911, -0.13778805],
            "stock1": [-0.13778805, 0.13090794],
        },
        index=["stock0", "stock1"],
    )

    pdt.assert_frame_equal(result, expected)


# =============================================================================
# TESTS OPTIMIZER
# =============================================================================


def test_OptimizerABC_not_implementhed_methods():
    class Foo(OptimizerABC):
        def serialize(self, port):
            return super().serialize(port)

        def optimize(self, port, target_return):
            return super().optimize(port, target_return)

    opt = Foo()
    with pytest.raises(NotImplementedError):
        opt.serialize(0)

    with pytest.raises(NotImplementedError):
        opt.optimize(0, 0)


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

    markowitz = Markowitz()
    result = markowitz.serialize(pf)

    expected_mu = pd.Series({"stock0": 46.121466, "stock1": 287.122362})
    expected_cov = pd.DataFrame(
        data={
            "stock0": [0.17805911, -0.13778805],
            "stock1": [-0.13778805, 0.13090794],
        },
        index=["stock0", "stock1"],
    )

    assert isinstance(result, dict)
    assert result.keys() == {"expected_returns", "cov_matrix", "weight_bounds"}
    pdt.assert_series_equal(expected_mu, result["expected_returns"])
    pdt.assert_frame_equal(expected_cov, result["cov_matrix"])
    assert result["weight_bounds"] == (0, 1)


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

    markowitz = Markowitz()
    result = markowitz.optimize(pf, target_return=1.0)
    expected_weights = pd.Series(
        data={"stock0": 0.45966836, "stock1": 0.54033164}, name="Weights"
    )

    # Assert everything is the same except for the weights
    assert result is not pf
    assert isinstance(result, Portfolio)
    pdt.assert_frame_equal(pf._df, result._df)

    assert isinstance(result.weights, pd.Series)
    pdt.assert_series_equal(result.weights, expected_weights)
