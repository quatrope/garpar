# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test entropy related functions."""


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.utils import entropy

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_proportion_shannon():
    """Test propotion Shannon."""
    arr = [
        [0.0454, 0.5558, 0.952, 0.0855],
        [0.3182, 0.4122, 0.8509, 0.9355],
        [0.3409, 0.5937, 0.7033, 0.731],
    ]

    ent = entropy.shannon(arr)
    np.testing.assert_allclose(
        ent, [0.88694495, 1.08693663, 1.0910435, 0.84711156]
    )


def test_proportion_shannon_with_window_warning():
    """Test propotion Shannon with window warning."""
    prices = [
        [0.0454, 0.5558, 0.952, 0.0855],
        [0.3182, 0.4122, 0.8509, 0.9355],
        [0.3409, 0.5937, 0.7033, 0.731],
    ]

    with pytest.warns(UserWarning):
        ent = entropy.shannon(prices, window_size=10)

    np.testing.assert_allclose(
        ent, [0.88694495, 1.08693663, 1.0910435, 0.84711156]
    )


def test_proportion_risso_window_size_None():
    """Test proportion of Risso for None window size."""
    prices = [
        [0.0454, 0.5558, 0.952, 0.0855],
        [0.3182, 0.4122, 0.8509, 0.9355],
        [0.3409, 0.5937, 0.7033, 0.731],
    ]

    with pytest.raises(ValueError):
        entropy.risso(prices, window_size=-1)


@pytest.mark.parametrize(
    "window_size, expected_entropy",
    [
        (3, [0.9994236129750128, 0.9967406675632541, 0.9984993168833038]),
        (5, [0.9953467238552131, 0.9900566703717419, 0.9903318205285124]),
        (7, [0.976574880029047, 0.9639653056779999, 0.976287579514779]),
    ],
)
def test_proportion_risso(risso_stocks_set, window_size, expected_entropy):
    """Test propotion Risso."""
    ss = risso_stocks_set(random_state=42, days=365, stocks=3)

    np.testing.assert_allclose(
        entropy.risso(ss.as_prices(), window_size=window_size),
        expected_entropy,
    )


def test_proportion_window_size_greater(risso_stocks_set):
    """Test propotion for a window size greater than days."""
    ss = risso_stocks_set(days=5)

    with pytest.raises(ValueError):
        entropy.risso(ss.as_prices(), window_size=10)


def test_yager_h1(risso_stocks_set):
    """Test yager entropy h1."""
    ss = risso_stocks_set(random_state=42)

    np.testing.assert_almost_equal(
        entropy.h_one(ss.weights), -3.3923344857821336
    )


def test_yager_h_inf(risso_stocks_set):
    """Test yager entropy for h->inf."""
    ss = risso_stocks_set(random_state=42)

    np.testing.assert_almost_equal(
        entropy.h_inf(ss.weights), 0.024377648363244075
    )


def test_yager_one(risso_stocks_set):
    """Test yager composition."""
    ss = risso_stocks_set(random_state=42)

    np.testing.assert_almost_equal(
        entropy.yager_one(ss.weights), 4.964303714641588
    )


def test_yager_inf(risso_stocks_set):
    """Test yager inf composition."""
    ss = risso_stocks_set(random_state=42)

    np.testing.assert_almost_equal(
        entropy.yager_inf(ss.weights), 0.875622351636756
    )
