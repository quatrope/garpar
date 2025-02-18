# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE
# =============================================================================
# DOCS
# =============================================================================

"""Test for garpar.utils.bunch module."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy

from garpar.utils import bunch

import pytest


# =============================================================================
# TEST Bunch
# =============================================================================


def test_Bunch_creation():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    assert len(md) == 1


def test_Bunch_creation_empty():
    md = bunch.Bunch("foo", {})
    assert len(md) == 0


def test_Bunch_key_notfound():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(KeyError):
        md["bravo"]


def test_Bunch_attribute_notfound():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(AttributeError):
        md.bravo


def test_Bunch_iter():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert list(iter(md)) == ["alfa"]


def test_Bunch_repr():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert repr(md) == "<foo {'alfa'}>"


def test_Bunch_dir():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert "alfa" in dir(md)


def test_Bunch_deepcopy():
    md = bunch.Bunch("foo", {"alfa": 1})
    md_c = copy.deepcopy(md)

    assert md is not md_c
    assert md._name == md_c._name  # string are inmutable never deep copy
    assert md._data == md_c._data and md._data is not md_c._data


def test_Bunch_copy():
    md = bunch.Bunch("foo", {"alfa": 1})
    md_c = copy.copy(md)

    assert md is not md_c
    assert md._name == md_c._name
    assert md._data == md_c._data and md._data is md_c._data


def test_Bunch_data_is_not_a_mapping():
    with pytest.raises(TypeError, match="Data must be some kind of mapping"):
        bunch.Bunch("foo", None)
