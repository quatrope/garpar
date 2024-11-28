# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

"""Package modules provided."""

__version__ = "0.1"

EPSILON = 1e-9

from . import datasets, io, optimize
from .core import StocksSet


__all__ = ["StocksSet", "datasets", "io", "optimize", "EPSILON"]
