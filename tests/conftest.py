# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Configuration of garpar tests

"""


# =============================================================================
# IMPORTS
# =============================================================================

from distutils.dist import Distribution
import functools


import numpy as np

import pytest

import garpar as gp

# =============================================================================
# CONSTANTS
# =============================================================================

DISTRIBUTION = {
    "levy-stable": gp.datasets.make_risso_levy_stable,
    "normal": gp.datasets.make_risso_normal,
    "uniform": gp.datasets.make_risso_uniform,
}

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def risso_portfolio():
    def make(*, distribution="normal", **kwargs):
        kwargs.setdefault("days", 5)
        kwargs.setdefault("stocks", 10)
        maker = DISTRIBUTION[distribution]
        return maker(**kwargs)

    return make


@pytest.fixture(scope="session")
def risso_portfolio_values(risso_portfolio):
    def make(*, distribution="normal", **kwargs):
        portfolio = risso_portfolio(distribution=distribution, **kwargs)
        return portfolio._df, portfolio._weights

    return make
