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

"""Test Utilities Accessor module."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

# =============================================================================
# DRISK TESTS
# =============================================================================


@pytest.fixture(scope="function")
def test_UtilitiesAccessor_ex_ante_tracking_error(risso_stocks_set):
    """Test UtilitiesAccessor ex ante tracking error."""
    ss = risso_stocks_set(random_state=42)
    expected = 0.0014537620394590928
    np.testing.assert_allclose(ss.utilities.ex_ante_tracking_error(), expected)


@pytest.fixture(scope="function")
def test_UtilitiesAccessor_ex_post_tracking_error(risso_stocks_set):
    """Test UtilitiesAccessor ex post tracking error."""
    ss = risso_stocks_set(random_state=42)
    expected = 0.2672203674826514
    np.testing.assert_allclose(ss.utilities.ex_post_tracking_error(), expected)


@pytest.fixture(scope="function")
def test_UtilitiesAccessor_portfolio_return(risso_stocks_set):
    """Test UtilitiesAccessor portfolio return."""
    ss = risso_stocks_set(random_state=42)
    expected = 0.15201221100419662
    np.testing.assert_allclose(ss.utilities.stocks_set_return(), expected)


@pytest.fixture(scope="function")
def test_UtilitiesAccessor_quadratic_utility(risso_stocks_set):
    """Test UtilitiesAccessor quadratic utility."""
    ss = risso_stocks_set(random_state=42)
    expected = 0.15243496338081428
    np.testing.assert_allclose(ss.utilities.quadratic_utility(), expected)
