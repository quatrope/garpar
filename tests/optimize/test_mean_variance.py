# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================


from garpar import Portfolio

from garpar.optimize.mean_variance import MVOptimizer, Markowitz

import pypfopt

from garpar import datasets

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# TESTS MV
# =============================================================================

def test_MVOptimizer_default_initialization():
    optimizer = MVOptimizer()
    assert optimizer.method == "max_sharpe"
    assert optimizer.weight_bounds == (0, 1)
    assert optimizer.market_neutral is False

def test_MVOptimizer_custom_initialization():
    optimizer = MVOptimizer(method="min_volatility", weight_bounds=(-1, 1))
    assert optimizer.method == "min_volatility"
    assert optimizer.weight_bounds == (-1, 1)

@pytest.mark.parametrize("method", pytest.METHODS)
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_calculate_weights_method_coerced(risso_portfolio, method, price_distribution):
    pf = risso_portfolio(random_state=42, distribution=price_distribution)
    optimizer = MVOptimizer(method=method)
    weights, meta = optimizer._calculate_weights(pf)
    assert len(weights) == len(pf.stocks)
    assert meta["name"] == method

@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_min_volatility(risso_portfolio, price_distribution):
    pf = risso_portfolio(random_state=42, distribution=price_distribution)
    optimizer = MVOptimizer(method="min_volatility")
    weights, meta = optimizer._calculate_weights(pf)
    assert len(weights) == len(pf.stocks)
    assert meta["name"] == "min_volatility"

@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_invalid_method(risso_portfolio, price_distribution):
    pf = risso_portfolio(random_state=42, distribution=price_distribution)
    optimizer = MVOptimizer(method="unknown_method")
    with pytest.raises(ValueError):
        optimizer._calculate_weights(pf)

@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_MVOptimizer_get_optimizer(risso_portfolio, price_distribution):
    pf = risso_portfolio(random_state=42, distribution=price_distribution)
    optimizer = MVOptimizer()
    assert type(optimizer._get_optimizer(pf)) == pypfopt.efficient_frontier.EfficientFrontier

@pytest.mark.parametrize(
    "volatiliy, price_distribution",
    [
        (0.03313144315467211, pytest.DISTRIBUTIONS['levy-stable']),
        (0.8509377578214843, pytest.DISTRIBUTIONS['normal']),
        (13.383798092382262, pytest.DISTRIBUTIONS['uniform'])
    ]
)
def test_MVOptimizer_coerce_volatility(volatiliy, price_distribution):
    pf = price_distribution(random_state=43)
    optimizer = MVOptimizer(method="max-sharpe")
    coerced_volatility = optimizer._coerce_target_volatility(pf)
    np.testing.assert_almost_equal(coerced_volatility, volatiliy, decimal=9)

# =============================================================================
# MARKOWITZ TEST
# =============================================================================

def test_Markowitz_optimize():
    pf = Portfolio.from_dfkws(
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

    # Tested method
    result = markowitz.optimize(pf)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45966836, "stock1": 0.54033164}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not pf
    pd.testing.assert_frame_equal(pf.as_prices(), result.as_prices())
    assert result.metadata.optimizer["target_return"] == 1

    assert isinstance(result.weights, pd.Series)
    pd.testing.assert_series_equal(result.weights, expected_weights)


def test_Markowitz_optimize_default_target_return():
    pf = Portfolio.from_dfkws(
        prices=pd.DataFrame(
            {
                "stock0": [1.11, 1.12, 1.10, 1.13, 1.18],
                "stock1": [10.10, 10.32, 10.89, 10.93, 11.05],
            },
        ),
        weights=[0.5, 0.5],
    )

    # Instance
    markowitz = Markowitz()

    # Tested method
    result = markowitz.optimize(pf)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45966836, "stock1": 0.54033164}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not pf
    pd.testing.assert_frame_equal(pf.as_prices(), result.as_prices())
    np.testing.assert_allclose(
        result.metadata.optimizer["target_return"], 0.0036730946
    )

    assert isinstance(result.weights, pd.Series)
    pd.testing.assert_series_equal(result.weights, expected_weights)
