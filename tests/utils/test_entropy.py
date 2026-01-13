# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2026
#   Lautaro Ebner,
#   Diego Gimenez,
#   Nadia Luczywo,
#   Juan Cabral,
#   and QuatroPe
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
        (3, [0.995950834702047, 0.9961331623543526, 0.998201315468035]),
        (5, [0.9932792483090254, 0.9890431488732914, 0.9901742466254664]),
        (7, [0.9757895066002318, 0.9652316216501301, 0.9768986095180278]),
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


def test_yager_one(risso_stocks_set):
    """Test yager composition."""
    ss = risso_stocks_set(random_state=42)

    np.testing.assert_almost_equal(
        entropy.yager_one(ss.weights), 4.975949018866288
    )


def test_yager_inf(risso_stocks_set):
    """Test yager inf composition."""
    ss = risso_stocks_set(random_state=42)

    np.testing.assert_almost_equal(
        entropy.yager_inf(ss.weights), 0.875622351636756
    )
