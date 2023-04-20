# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for garpar."""

# =============================================================================
# IMPORTS
# =============================================================================


from . import entropy, mabc, scalers

# export skcutils as own utils
from skcriteria.utils import Bunch as _SKCBunch, df_temporal_header, accabc


# custom bunch method


class Bunch(_SKCBunch):
    def to_dict(self):
        import copy  # noqa

        return copy.deepcopy(self._data)


# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "accabc",
    "df_temporal_header",
    "Bunch",
    "unique_names",
    "mabc",
    "scalers",
    "entropy",
]
