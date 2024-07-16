# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================


from garpar import optimize, Portfolio


import numpy as np

import pandas as pd

import pytest


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


def test_OptimizerABC__calculate_weights_not_implementhed(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    class FooOptimizer(optimize.OptimizerABC):
        def _calculate_weights(self, pf):
            return super()._calculate_weights(pf)

    with pytest.raises(NotImplementedError):
        FooOptimizer().optimize(pf)


# =============================================================================
# MARKOWIKS TEST
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
    markowitz = optimize.Markowitz(target_return=1.0)

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
    markowitz = optimize.Markowitz()

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


# =============================================================================
# BLACK LITTERMAN
# =============================================================================


def test_BlackLitterman_optimize():
    pf = Portfolio.from_dfkws(
        prices=pd.DataFrame(
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
    bl = optimize.BlackLitterman(prior=prior, absolute_views=viewdict)

    # Tested method
    result = bl.optimize(pf)

    # Expectations
    expected_weights = pd.Series(
        data={"stock0": 0.45157882, "stock1": 0.54842117}, name="Weights"
    )
    expected_weights.index.name = "Stocks"

    # Assert everything is the same except for the weights
    assert result is not pf
    pd.testing.assert_frame_equal(pf.as_prices(), result.as_prices())

    assert isinstance(result.weights, pd.Series)
    pd.testing.assert_series_equal(result.weights, expected_weights)
