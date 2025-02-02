# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test scalers."""

# =============================================================================
# IMPORTS
# =============================================================================

from garpar.utils import scalers

import numpy as np


# =============================================================================
# TESTS
# =============================================================================


def test_proportion_scaler():
    """Test propotion scaler."""
    arr = [10, 20, 70]
    scaled = scalers.proportion_scaler(arr)
    np.testing.assert_allclose(scaled, [0.1, 0.2, 0.7])


def test_minmax_scaler():
    """Test minmax scaler."""
    arr = [10, 20, 70]
    scaled = scalers.minmax_scaler(arr)
    np.testing.assert_allclose(scaled, [0, 0.166667, 1], rtol=1e-5)


def test_max_scaler():
    """Test max scaler."""
    arr = [10, 20, 70]
    scaled = scalers.max_scaler(arr)
    np.testing.assert_allclose(scaled, [0.14285714, 0.28571429, 1.0])


def test_standar_scaler():
    """Test standar scaler."""
    arr = [10, 20, 70]
    scaled = scalers.standar_scaler(arr)
    np.testing.assert_allclose(scaled, [-0.88900089, -0.50800051, 1.3970014])
