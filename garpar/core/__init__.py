# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Core functionality for Garpar project.

The core module provides a main class named StocksSet that represents both
a portfolio and a market.

"""

# =============================================================================
# IMPORTS
# =============================================================================


from .stocks_set import StocksSet, mkss


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = ["StocksSet", "mkss"]
