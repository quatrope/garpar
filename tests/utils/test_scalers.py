# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.utils import scalers

import numpy as np


# =============================================================================
# TESTS
# =============================================================================


def test_proportion_scaler():
    arr = [10, 20, 70]
    scaled = scalers.proportion_scaler(arr)
    np.testing.assert_allclose(scaled, [0.1, 0.2, 0.7])


def test_minmax_scaler():
    arr = [10, 20, 70]
    scaled = scalers.minmax_scaler(arr)
    np.testing.assert_allclose(scaled, [0, 0.166667, 1], rtol=1e-5)


def test_max_scaler():
    arr = [10, 20, 70]
    scaled = scalers.max_scaler(arr)
    np.testing.assert_allclose(scaled, [0.14285714, 0.28571429, 1.0])


def test_standar_scaler():
    arr = [10, 20, 70]
    scaled = scalers.standar_scaler(arr)
    np.testing.assert_allclose(scaled, [-0.88900089, -0.50800051, 1.3970014])
