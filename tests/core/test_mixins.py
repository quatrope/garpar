# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from garpar import StocksSet
from garpar.core._mixins import CoercerMixin

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize("ereturn", ["capm", "mah", "emah"])
def test_CoercerMixin_coerce_expected_returns(risso_stocks_set, ereturn):
    ss = risso_stocks_set(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _ss = ss

    coercer = Coercer()

    result = coercer.coerce_expected_returns(ereturn, {})
    expected = ss.ereturns(ereturn).to_numpy()

    np.testing.assert_array_equal(result, expected)


def test_CoercerMixin_coerce_weights_None(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _ss = ss

    coercer = Coercer()

    result = coercer.coerce_weights(None)

    cols = len(ss.stocks)
    expected = np.full(cols, 1.0 / cols, dtype=float)

    np.testing.assert_array_equal(result, expected)


def test_CoercerMixin_coerce_weights_another_ss(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _ss = ss

    coercer = Coercer()

    ss_bench = StocksSet.from_dfkws(ss.as_prices(), weights=[0.9, 0.1])
    expected = ss_bench.weights.to_numpy()

    result = coercer.coerce_weights(ss_bench)

    np.testing.assert_array_equal(result, expected)


def test_CoercerMixin_coerce_weights_bad_ss(risso_stocks_set):
    ss = risso_stocks_set(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _ss = ss

    coercer = Coercer()

    prices = ss.as_prices()
    prices.columns = ["S0", "SB"]
    ss_bench = StocksSet.from_dfkws(prices, weights=[0.9, 0.1])

    with pytest.raises(KeyError):
        coercer.coerce_weights(ss_bench)


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
def test_CoercerMixin_coerce_covariance_matrix(risso_stocks_set, cov_matrix):
    ss = risso_stocks_set(random_state=42, stocks=2)

    class Coercer(CoercerMixin):
        _ss = ss

    coercer = Coercer()

    result = coercer.coerce_covariance_matrix(cov_matrix, {})
    expected = ss.cov(cov_matrix).to_numpy()

    np.testing.assert_array_equal(result, expected)
