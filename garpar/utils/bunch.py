# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Bunch util for Garpar project."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy
from collections.abc import Mapping


# =============================================================================
# DOC INHERITANCE
# =============================================================================


class Bunch(Mapping):
    """Container object exposing keys as attributes.

    Concept based on the sklearn.utils.Bunch.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> b = SKCBunch("data", {"a": 1, "b": 2})
    >>> b
    data({a, b})
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, name, data):
        if not isinstance(data, Mapping):
            raise TypeError("Data must be some kind of mapping")
        self._name = str(name)
        self._data = data

    def __getitem__(self, k):
        """x.__getitem__(y) <==> x[y]."""
        return self._data[k]

    def __getattr__(self, a):
        """x.__getattr__(y) <==> x.y."""
        try:
            return self._data[a]
        except KeyError:
            raise AttributeError(a)

    def __copy__(self):
        """x.__copy__() <==> copy.copy(x)."""
        cls = type(self)
        return cls(str(self._name), data=self._data)

    def __deepcopy__(self, memo):
        """x.__deepcopy__() <==> copy.copy(x)."""
        # extract the class
        cls = type(self)

        # make the copy but without the data
        clone = cls(name=str(self._name), data={})

        # store in the memo that clone is copy of self
        # https://docs.python.org/3/library/copy.html
        memo[id(self)] = clone

        # now we copy the data
        clone._data = copy.deepcopy(self._data, memo)

        return clone

    def __iter__(self):
        """x.__iter__() <==> iter(x)."""
        return iter(self._data)

    def __len__(self):
        """x.__len__() <==> len(x)."""
        return len(self._data)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        content = repr(set(self._data)) if self._data else "{}"
        return f"<{self._name} {content}>"

    def __dir__(self):
        """x.__dir__() <==> dir(x)."""
        return super().__dir__() + list(self._data)

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
