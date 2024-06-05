# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================


# from garpar import datasets

# import numpy as np

# import pytest


# =============================================================================
# TESTS
# =============================================================================

# @pytest.mark.parametrize(
#     "imputation, mean, std",
#     [
#         ("ffill", 136.60173850574714, 52.66562555912109),
#         ("pad", 136.60173850574714, 52.66562555912109),
#         ("bfill", 136.60018530798706, 52.665682290838745),
#         ("backfill", 136.60018530798706, 52.665682290838745),
#         (42, 136.56374963159448, 52.82468365235412),
#     ],
# )
# def test_load_merval2021_2022(imputation, mean, std):
#     pf = datasets.load_merval2021_2022(imputation)
#     np.testing.assert_almost_equal(pf.prices.mean().mean(), mean, decimal=13)
#     np.testing.assert_almost_equal(pf.prices.std().mean(), std, decimal=13)
