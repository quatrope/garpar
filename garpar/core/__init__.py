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


"""Core subpackage of Garpar project.

This subpackage is centered around the StocksSet class, which represents both a
portfolio and a market in a single structure. It provides accessors to compute
various properties, graphs, and metrics of a portfolio.

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
