# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# This code was ripped of from scikit-criteria on 13-apr-2023.
# https://github.com/quatrope/scikit-criteria/blob/ec63c/skcriteria/utils/bunch.py
# Util this point the copyright is
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.


# =============================================================================
# DOCS
# =============================================================================

"""Utility to achieve unique names for a collection of objects."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import Counter


# =============================================================================
# FUNCTIONS
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
