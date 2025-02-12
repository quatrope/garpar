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
the StocksSet class, accessors of that class and related functions to it.

Key Features:
    - Correlation and covariance analysis
    - Diversification metrics
    - Entropy-based analysis
    - Expected returns estimation
    - Market data handling and validation
    - Portfolio/market construction and rebalancing
    - Portfolio/market visualization tools
    - Price-related data and methods
    - Risk metrics calculation (variance, VaR, etc.)
    - Utility metrics calculation

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .stocks_set import StocksSet, mkss

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["StocksSet", "mkss"]
