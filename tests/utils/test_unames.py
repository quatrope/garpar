# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE
# =============================================================================
# DOCS
# =============================================================================

"""Test for garpar.utils.unames"""


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.utils import unames

import pytest


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_unique_names():
    names, elements = ["foo", "faa"], [0, 1]
    result = dict(unames.unique_names(names=names, elements=elements))
    expected = {"foo": 0, "faa": 1}
    assert result == expected


def test_unique_names_with_duplticates():
    names, elements = ["foo", "foo"], [0, 1]
    result = dict(unames.unique_names(names=names, elements=elements))
    expected = {"foo_1": 0, "foo_2": 1}
    assert result == expected


def test_unique_names_with_different_len():
    names, elements = ["foo", "foo"], [0]
    with pytest.raises(ValueError):
        unames.unique_names(names=names, elements=elements)
