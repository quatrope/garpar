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

"""Test Optimize base module."""

# =============================================================================
# IMPORTS
# =============================================================================

from garpar.optimize.opt_base import MeanVarianceFamilyMixin, OptimizerABC

import pytest


# =============================================================================
# DIVERSIFICATION TESTS
# =============================================================================


def test_OptimizerABC__calculate_weights_not_implementhed(risso_stocks_set):
    """Test for OptimizerABC when calculate weights are not implemented."""
    ss = risso_stocks_set(random_state=42, stocks=2)

    class FooOptimizer(MeanVarianceFamilyMixin, OptimizerABC):
        def _calculate_weights(self, ss):
            return super()._calculate_weights(ss)

    with pytest.raises(NotImplementedError):
        FooOptimizer().optimize(ss)


def test_optimizerabc_family_not_string():
    """Test that an error is raised if 'family' is not a string."""
    with pytest.raises(
        TypeError,
        match="'InvalidOptimizer.family' must be redefined as string",
    ):

        class InvalidOptimizer(OptimizerABC):
            family = 123  # Not a string


def test_optimizerabc_family_undefined():
    """Test that an error is raised if 'family' is not defined."""
    with pytest.raises(
        TypeError,
        match="'UndefinedFamilyOptimizer.family' must be redefined as string",
    ):

        class UndefinedFamilyOptimizer(OptimizerABC):
            pass


def test_optimizerabc_family_valid():
    """Test that no error is raised when 'family' is a valid string."""

    class ValidOptimizer(OptimizerABC):
        family = "MeanVariance"

    assert ValidOptimizer.family == "MeanVariance"


def test_OptimizerABC_get_family():
    """Test OptimizerABC family."""

    class TestMeanVarianceFamily(MeanVarianceFamilyMixin, OptimizerABC):
        pass

    assert TestMeanVarianceFamily.get_optimizer_family() == "mean-variance"
