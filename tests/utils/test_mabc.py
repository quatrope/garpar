# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test ModelABC and hyperparameters."""


# =============================================================================
# IMPORTS
# =============================================================================

import attr

from garpar.utils import mabc

# =============================================================================
# TESTS
# =============================================================================


def test_hparam():
    """Test hparam."""
    result = mabc.hparam(default=1)
    assert result.metadata[mabc.HPARAM_METADATA_FLAG]
    assert result.kw_only
    assert result._default == 1


def test_mproperty():
    """Test mproperty."""
    result = mabc.mproperty()
    assert result.metadata[mabc.MPROPERTY_METADATA_FLAG]
    assert result.init is False
    assert result._default is attr.NOTHING


def test_ModelABC():
    """Test ModelABC."""

    class TestModel(mabc.ModelABC):
        param = mabc.hparam(default=1)
        prop = mabc.mproperty()

        @prop.default
        def _prop_default(self):
            return self.param + 1

    model = TestModel(param=2)
    assert model.prop == 3
    assert repr(model) == "TestModel(param=2)"
