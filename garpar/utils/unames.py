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

"""Unique names util for Garpar project."""


# =============================================================================
# IMPORTS
# =============================================================================

from collections import Counter


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
