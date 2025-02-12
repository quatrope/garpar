# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================


"""Core module.

This module contains the core functionality of the Garpar project, including
the StocksSet class and related functions to it.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .stocks_set import StocksSet, mkss

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["StocksSet", "mkss"]
