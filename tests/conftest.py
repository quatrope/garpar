# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Configuration of garpar tests."""


# =============================================================================
# IMPORTS
# =============================================================================

import garpar as gp

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

DISTRIBUTIONS = {
    "levy-stable": gp.datasets.make_risso_levy_stable,
    "normal": gp.datasets.make_risso_normal,
    "uniform": gp.datasets.make_risso_uniform,
}

METHODS = [
    "min_volatility",
    "max_sharpe",
    "max_quadratic_utility",
    "efficient_risk",
    "efficient_return",
]

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def risso_portfolio():
    def make(*, distribution="normal", random_state=None, **kwargs):
        maker = DISTRIBUTIONS[distribution]
        random_state = np.random.default_rng(random_state)
        kwargs.setdefault("days", 5)
        kwargs.setdefault("stocks", 10)

        pf = maker(random_state=random_state, **kwargs)

        weights = random_state.random(size=len(pf.weights))
        pf = pf.copy(weights=weights)

        return pf

    return make


@pytest.fixture(scope="session")
def risso_portfolio_values(risso_portfolio):
    def make(*, distribution="normal", **kwargs):
        portfolio = risso_portfolio(distribution=distribution, **kwargs)
        return portfolio._df, portfolio._weights

    return make


# =============================================================================
# CONFIGURATIONS
# =============================================================================


def pytest_configure():
    pytest.DISTRIBUTIONS = DISTRIBUTIONS
    pytest.METHODS = METHODS
