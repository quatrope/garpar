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
# =============================================================================
# DOCS
# =============================================================================

"""Test for garpar.utils.accabc module."""


# =============================================================================
# IMPORTS
# =============================================================================


from garpar.utils import AccessorABC

import numpy as np

import pytest

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_AccessorABC():
    class FooAccessor(AccessorABC):
        _default_kind = "zaraza"

        def __init__(self, v):
            self._v = v

        def zaraza(self):
            return self._v

    acc = FooAccessor(np.random.random())
    assert acc("zaraza") == acc.zaraza() == acc()


def test_AccessorABC_no__default_kind():
    with pytest.raises(TypeError):

        class FooAccessor(AccessorABC):
            pass

    with pytest.raises(TypeError):
        AccessorABC()


def test_AccessorABC_invalid_kind():
    class FooAccessor(AccessorABC):
        _default_kind = "zaraza"

        def __init__(self):
            self.dont_work = None

        def _zaraza(self):
            pass

    acc = FooAccessor()

    with pytest.raises(ValueError):
        acc("_zaraza")

    with pytest.raises(ValueError):
        acc("dont_work")
