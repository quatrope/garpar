# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
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
from skcriteria.utils import (
    Bunch as _SKCBunch,
    df_temporal_header,
    accabc,
    unique_names,
)


# custom bunch method


class Bunch(_SKCBunch):
    """
    Container object exposing keys as attributes.

    The Bunch class extends the _SKCBunch class, allowing for deep copying of its
    data attribute to a dictionary format.

    Methods
    -------
    to_dict():
        Returns a deep copy of the _data attribute as a dictionary.
    """

    def to_dict(self):
        """
        Convert the Bunch object to a dictionary.

        This method performs a deep copy of the _data attribute, ensuring that 
        the original data remains unchanged.

        Returns
        -------
        dict
            A deep copy of the _data attribute.

        Examples
        --------
        >>> bunch = Bunch()
        >>> bunch._data = {'key1': 'value1', 'key2': 'value2'}
        >>> dict_data = bunch.to_dict()
        >>> print(dict_data)
        {'key1': 'value1', 'key2': 'value2'}
        """
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
