from typing import Type

import inspect


class AccessorABC:

    _DEFAULT_KIND = None
    _FORBIDDEN_METHODS = []

    def __init_subclass__(cls):
        if cls._DEFAULT_KIND is None:
            raise TypeError("You must redefine class Attribute _DEFAULT_KIND")

    def __call__(self, kind=None, **kwargs):

        kind = self._DEFAULT_KIND if kind is None else kind

        if kind.startswith("_") or kind in self._FORBIDDEN_METHODS:
            raise ValueError(f"invalid kind name '{kind}'")

        method = getattr(self, kind, None)
        if not callable(method):
            raise ValueError(f"invalid kind name '{kind}'")

        return method(**kwargs)

    def __repr__(self):
        return f"{type(self).__name__}(DEFAULT_KIND={self._DEFAULT_KIND})"
