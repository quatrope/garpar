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

from garpar.optimize import mean_variance

from garpar import datasets

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# MARKOWITS TEST
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
    markowitz = mean_variance.Markowitz(target_return=1.0)

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
    markowitz = mean_variance.Markowitz()

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

