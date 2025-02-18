# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Accessor abstract base class for Garpar project."""


# =============================================================================
# IMPORTS
# =============================================================================

import abc


# =============================================================================
# CONSTANTS
# =============================================================================

# This constans are used to mark a class attribute as abstract, and prevet an
# instantiaiton of a class
_ABSTRACT = property(abc.abstractmethod(lambda: ...))


# =============================================================================
# CONSTANTS
# =============================================================================


class AccessorABC(abc.ABC):
    """Generalization of the accessor idea for use in Garpar.

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
