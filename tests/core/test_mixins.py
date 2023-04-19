# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar import Portfolio
from garpar.core._mixins import CoercerMixin

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize("ereturn", ["capm", "mah", "emah"])
def test_CoercerMixin_coerce_expected_returns(risso_portfolio, ereturn):
    pf = risso_portfolio(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _pf = pf

    coercer = Coercer()

    result = coercer.coerce_expected_returns(ereturn, {})
    expected = pf.ereturns(ereturn).to_numpy()

    np.testing.assert_array_equal(result, expected)


def test_CoercerMixin_coerce_weights_None(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _pf = pf

    coercer = Coercer()

    result = coercer.coerce_weights(None)

    cols = len(pf.stocks)
    expected = np.full(cols, 1.0 / cols, dtype=float)

    np.testing.assert_array_equal(result, expected)


def test_CoercerMixin_coerce_weights_another_pf(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _pf = pf

    coercer = Coercer()

    pf_bench = Portfolio.from_dfkws(pf.as_prices(), weights=[0.9, 0.1])
    expected = pf_bench.weights.to_numpy()

    result = coercer.coerce_weights(pf_bench)

    np.testing.assert_array_equal(result, expected)


def test_CoercerMixin_coerce_weights_bad_pf(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _pf = pf

    coercer = Coercer()

    prices = pf.as_prices()
    prices.columns = ["S0", "SB"]
    pf_bench = Portfolio.from_dfkws(prices, weights=[0.9, 0.1])

    with pytest.raises(KeyError):
        coercer.coerce_weights(pf_bench)


@pytest.mark.parametrize(
    "cov_matrix",
    [
        "sample_cov",
        "exp_cov",
        "semi_cov",
        "ledoit_wolf_cov",
        "oracle_approximating_cov",
    ],
)
def test_CoercerMixin_coerce_covariance_matrix(risso_portfolio, cov_matrix):
    pf = risso_portfolio(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _pf = pf

    coercer = Coercer()

    result = coercer.coerce_covariance_matrix(cov_matrix, {})
    expected = pf.cov(cov_matrix).to_numpy()

    np.testing.assert_array_equal(result, expected)
