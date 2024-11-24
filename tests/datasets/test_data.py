# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================


from garpar import datasets

import numpy as np

import pytest


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize(
    "imputation, mean, std",
    [
        ("ffill", 62.04622807409393, 70.3072133444952),
        ("pad", 62.04622807409393, 70.3072133444952),
        ("bfill", 62.045251347568424, 70.30657769357053),
        ("backfill", 62.045251347568424, 70.30657769357053),
        (42, 62.04957287292654, 70.40979087943586),
    ],
)
def test_load_MERVAL(imputation, mean, std):
    ss = datasets.load_MERVAL(imputation)
    np.testing.assert_almost_equal(ss.prices.mean().mean(), mean, decimal=13)
    np.testing.assert_almost_equal(ss.prices.std().mean(), std, decimal=13)
