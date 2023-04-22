# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.core import div_acc

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# DRISK TESTS
# =============================================================================


def test_UtilitiesAccessor_ex_ante_tracking_error(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 0.008737194
    np.testing.assert_allclose(pf.utilities.ex_ante_tracking_error(), expected)


def test_UtilitiesAccessor_ex_post_tracking_error(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 0.026393705
    np.testing.assert_allclose(pf.utilities.ex_post_tracking_error(), expected)


def test_UtilitiesAccessor_portfolio_return(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = -0.518791583
    np.testing.assert_allclose(pf.utilities.portfolio_return(), expected)


def test_UtilitiesAccessor_quadratic_utility(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = -0.516094918
    np.testing.assert_allclose(pf.utilities.quadratic_utility(), expected)