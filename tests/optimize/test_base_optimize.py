# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================


from garpar import Portfolio

from garpar.optimize import opt_base as base

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


def test_OptimizerABC__calculate_weights_not_implementhed(risso_portfolio):
    pf = risso_portfolio(random_state=42, stocks=2)

    class FooOptimizer(base.MeanVarianceFamilyMixin, base.OptimizerABC):
        def _calculate_weights(self, pf):
            return super()._calculate_weights(pf)

    with pytest.raises(NotImplementedError):
        FooOptimizer().optimize(pf)
