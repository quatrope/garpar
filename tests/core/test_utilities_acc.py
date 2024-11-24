# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# DRISK TESTS
# =============================================================================


def test_UtilitiesAccessor_ex_ante_tracking_error(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = 0.008737194
    np.testing.assert_allclose(ss.utilities.ex_ante_tracking_error(), expected)


def test_UtilitiesAccessor_ex_post_tracking_error(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = 0.026393705
    np.testing.assert_allclose(ss.utilities.ex_post_tracking_error(), expected)


def test_UtilitiesAccessor_sotcks_set_return(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = -0.518791583
    np.testing.assert_allclose(ss.utilities.sotcks_set_return(), expected)


def test_UtilitiesAccessor_quadratic_utility(risso_stocks_set):
    ss = risso_stocks_set(random_state=42)
    expected = -0.516094918
    np.testing.assert_allclose(ss.utilities.quadratic_utility(), expected)
