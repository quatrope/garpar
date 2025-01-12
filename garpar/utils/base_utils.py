# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Base utils for Garpar project."""


# =============================================================================
# IMPORTS
# =============================================================================

import abc
import contextlib
import copy
from collections import Counter
from collections.abc import Mapping
import inspect


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


# =============================================================================
# ACESSOR ABC
# =============================================================================

# This constans are used to mark a class attribute as abstract, and prevet an
# instantiaiton of a class
_ABSTRACT = property(abc.abstractmethod(lambda: ...))


class AccessorABC(abc.ABC):
    """Generalization of the accessor idea for use in scikit-criteria.

    Instances of this class are callable and accept as the first
    parameter 'kind' the name of a method to be executed followed by all the
    all the parameters of this method.

    If 'kind' is None, the method defined in the class variable
    '_default_kind_kind' is used.

    The last two considerations are that 'kind', cannot be a private method and
    that all subclasses of the method and that all AccessorABC subclasses have
    to redefine '_default_kind'.

    """

    #: Default method to execute.
    _default_kind = _ABSTRACT

    def __init_subclass__(cls):
        """Validate the creation of a subclass."""
        if cls._default_kind is _ABSTRACT:
            raise TypeError(f"{cls!r} must define a _default_kind")

    def __call__(self, kind=None, **kwargs):
        """x.__call__() <==> x()."""
        kind = self._default_kind if kind is None else kind

        if kind.startswith("_"):
            raise ValueError(f"invalid kind name '{kind}'")

        method = getattr(self, kind, None)
        if not callable(method):
            raise ValueError(f"Invalid kind name '{kind}'")

        return method(**kwargs)


# =============================================================================
# TEMPORAL HEADER
# =============================================================================


@contextlib.contextmanager
def df_temporal_header(df, header, name=None):
    """Temporarily replaces a DataFrame columns names.

    Optionally also assign another name to the columns.

    Parameters
    ----------
    header : sequence
        The new names of the columns.
    name : str or None (default None)
        New name for the index containing the columns in the DataFrame. If
        'None' the original name of the columns present in the DataFrame is
        preserved.

    """
    original_header = df.columns
    original_name = original_header.name

    name = original_name if name is None else name
    try:
        df.columns = header
        df.columns.name = name
        yield df
    finally:
        df.columns = original_header
        df.columns.name = original_name


# =============================================================================
# UNIQUE NAMES
# =============================================================================


def unique_names(*, names, elements):
    """Generate names unique name.

    Parameters
    ----------
    elements: iterable of size n
        objects to be named
    names: iterable of size n
        names candidates

    Returns
    -------
    list of tuples:
        Returns a list where each element is a tuple.
        Each tuple contains two elements: The first element is the unique name
        of the second is the named object.

    """
    # Based on sklearn.pipeline._name_estimators
    if len(names) != len(elements):
        raise ValueError("'names' and 'elements' must have the same length")

    names = list(reversed(names))
    elements = list(reversed(elements))

    name_count = {k: v for k, v in Counter(names).items() if v > 1}

    named_elements = []
    for name, step in zip(names, elements):
        count = name_count.get(name, 0)
        if count:
            name_count[name] = count - 1
            name = f"{name}_{count}"

        named_elements.append((name, step))

    named_elements.reverse()

    return named_elements
