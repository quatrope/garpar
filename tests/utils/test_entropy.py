# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


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
    prices = [
        [0.0454, 0.5558, 0.952, 0.0855],
        [0.3182, 0.4122, 0.8509, 0.9355],
        [0.3409, 0.5937, 0.7033, 0.731],
    ]

    with pytest.raises(ValueError):
        entropy.risso(prices, window_size=-1)
