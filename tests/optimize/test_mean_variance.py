# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test mean variance module."""

# =============================================================================
# IMPORTS
# =============================================================================


from garpar import StocksSet
from garpar.optimize.mean_variance import MVOptimizer, Markowitz

import numpy as np

import pandas as pd

import pypfopt

import pytest

# =============================================================================
# TESTS MV
# =============================================================================


def test_MVOptimizer_default_initialization():
    """Test MVOptimizer default initialization."""
    optimizer = MVOptimizer()
    assert optimizer.model == "max_sharpe"
    assert optimizer.weight_bounds == (0, 1)
    assert optimizer.market_neutral is False


def test_MVOptimizer_custom_initialization():
    """Test MVOptimizer custom initialization."""
    optimizer = MVOptimizer(model="min_volatility", weight_bounds=(-1, 1))
    assert optimizer.model == "min_volatility"
    assert optimizer.weight_bounds == (-1, 1)


@pytest.mark.parametrize("model", pytest.MODELS)
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_calculate_weights_model_coerced(
    risso_stocks_set, model, price_distribution
):
    """Test MVOptimizer calculate weights model coerced."""
    ss = risso_stocks_set(random_state=50, distribution=price_distribution)
    optimizer = MVOptimizer(model=model)
    weights, meta = optimizer._calculate_weights(ss)

    assert len(weights) == len(ss.stocks)
    assert meta["name"] == model


@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_min_volatility(risso_stocks_set, price_distribution):
    """Test MVOptimizer min volatility model."""
    ss = risso_stocks_set(random_state=42, distribution=price_distribution)
    optimizer = MVOptimizer(model="min_volatility")
    weights, meta = optimizer._calculate_weights(ss)
    assert len(weights) == len(ss.stocks)
    assert meta["name"] == "min_volatility"


@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_invalid_model(risso_stocks_set, price_distribution):
    """Test MVOptimizer invalid model."""
    risso_stocks_set(random_state=42, distribution=price_distribution)
    with pytest.raises(ValueError):
        MVOptimizer(model="unknown_model")


@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_get_optimizer(risso_stocks_set, price_distribution):
    """Test MVOptimizer get optimizer."""
    ss = risso_stocks_set(random_state=42, distribution=price_distribution)
    optimizer = MVOptimizer()
    assert (
        type(optimizer._get_optimizer(ss))
        == pypfopt.efficient_frontier.EfficientFrontier
    )


@pytest.mark.parametrize(
    "volatiliy, price_distribution",
    [
        (0.0005845375959651712, pytest.DISTRIBUTIONS["levy-stable"]),
        (0.009733987973370875, pytest.DISTRIBUTIONS["normal"]),
        (0.1473086199704777, pytest.DISTRIBUTIONS["uniform"]),
    ],
)
def test_MVOptimizer_coerce_risk(volatiliy, price_distribution):
    """Test MVOptimizer coerce risk."""
    ss = price_distribution(random_state=43)
    optimizer = MVOptimizer(model="max_sharpe")
    coerced_risk = optimizer._coerce_target_risk(ss)
    np.testing.assert_almost_equal(coerced_risk, volatiliy, decimal=9)


# =============================================================================
# MARKOWITZ TEST
# =============================================================================


def test_Markowitz_optimize():
    """Test Markowitz optimizer."""
    ss = StocksSet.from_prices(
        prices=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            }
        ),
        weights=[0.5, 0.5],
    )

    # Instance
    markowitz = Markowitz(target_return=1.0)

    # Tested model
    result = markowitz.optimize(ss)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45966836, "stock1": 0.54033164}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not ss
    pd.testing.assert_frame_equal(ss.as_prices(), result.as_prices())
    assert result.metadata.optimizer["target_return"] == 1

    assert isinstance(result.weights, pd.Series)
    pd.testing.assert_series_equal(result.weights, expected_weights)


def test_Markowitz_optimize_default_target_risk():
    """Test Markowitz optimize default target return."""
    ss = StocksSet.from_prices(
        prices=pd.DataFrame(
            {
                "stock0": [
                    100.000000,
                    100.134675,
                    100.372761,
                    100.346268,
                    100.194167,
                    100.294895,
                ],
                "stock1": [
                    100.000000,
                    100.448529,
                    100.397479,
                    100.354087,
                    100.377197,
                    100.421145,
                ],
            }
        ),
        weights=[1.0, 1.0],
    )

    # Instance
    markowitz = Markowitz()

    # Tested model
    result = markowitz.optimize(ss)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.691589, "stock1": 0.308411}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not ss
    pd.testing.assert_frame_equal(ss.as_prices(), result.as_prices())
    np.testing.assert_allclose(
        result.metadata.optimizer["target_risk"], 0.021126518730178588
    )

    assert isinstance(result.weights, pd.Series)
    pd.testing.assert_series_equal(result.weights, expected_weights)


def test_Markowitz_optimize_default_target_return():
    """Test Markowitz optimize default target return."""
    ss = StocksSet.from_prices(
        prices=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10],
                "stock1": [10.10, 10.32, 10.89],
            },
        ),
        weights=[0.5, 0.5],
    )

    # Instance
    markowitz = Markowitz(target_return=0.01)

    # Tested model
    result = markowitz.optimize(ss)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.554581, "stock1": 0.445419}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not ss
    pd.testing.assert_frame_equal(ss.as_prices(), result.as_prices())
    np.testing.assert_allclose(
        result.metadata.optimizer["target_return"], 0.01
    )

    assert isinstance(result.weights, pd.Series)
    pd.testing.assert_series_equal(result.weights, expected_weights)
