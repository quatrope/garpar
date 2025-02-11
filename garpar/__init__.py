# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Generation and analysis of artificial and real portfolio returns.

A comprehensive toolset for analyzing and managing financial portfolios/markets
through the StocksSet class. Provides functionality for portfolio/market
optimization, risk assessment, and performance analysis.

Key Features:
    - Portfolio/market construction and rebalancing
    - Risk metrics calculation (variance, VaR, etc.)
    - Expected returns estimation
    - Correlation and covariance analysis
    - Diversification metrics
    - Portfolio/market visualization tools
    - Market data handling and validation
    - Entropy-based analysis

"""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib

from . import constants, datasets, garpar_io, optimize
from .core import StocksSet, mkss


# =============================================================================
# METADATA
# =============================================================================

__version__ = importlib.metadata.version("garpar")


__all__ = [
    "StocksSet",
    "datasets",
    "garpar_io",
    "optimize",
    "mkss",
    "constants",
]
