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
# DIVERSIFICATION TESTS
# =============================================================================


def test_DiversificationAccessor_ratio(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 0.019438624174371914
    np.testing.assert_allclose(pf.div.ratio(), expected)


def test_DiversificationAccessor_mrc(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=3)

    expected = pd.Series(
        [0.0030858316004040128, 0.012723758367798243, 0.006026050378237068],
        name="MRC",
        index=["S0", "S1", "S2"],
    )
    expected.index.name = "Stocks"

    pd.testing.assert_series_equal(
        pf.div.mrc(), expected, rtol=1e-05, atol=1e-06
    )


def test_DiversificationAccessor_pdi(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 3.0459243730151204
    np.testing.assert_allclose(pf.div.pdi(), expected)


def test_DiversificationAccessor_zheng_entropy(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 2.302585
    np.testing.assert_allclose(pf.div.zheng_entropy(), expected)

def test_DiversificationAccessor_cross_entropy(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 0
    np.testing.assert_allclose(pf.div.cross_entropy(), expected)


def test_DiversificationAccessor_ke_zang_entropy(risso_portfolio):
    pf = risso_portfolio(random_state=42)
    expected = 2.3133717
    np.testing.assert_allclose(pf.div.ke_zang_entropy(), expected)
