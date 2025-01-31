# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

"""Package modules provided."""

__version__ = "1.0"

__EPSILON__ = 1e-9

from . import datasets, garpar_io, optimize
from .core import StocksSet, mkss


__all__ = [
    "StocksSet",
    "datasets",
    "garpar_io",
    "optimize",
    "mkss",
    "__EPSILON__",
]
