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

Available subpackages
---------------------
core
    Core functionality for creating and analyzing StocksSet objects.
datasets
    Functionality to create and manipulate StocksSet objects.
optimize
    Optimization models for StocksSets objects.

Functions:
    mkss: Function for creating StocksSet objects.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib

from . import constants, core, datasets, garpar_io, optimize
from .core import StocksSet, mkss

# =============================================================================
# METADATA
# =============================================================================

__version__ = importlib.metadata.version("garpar")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "constants",
    "core",
    "datasets",
    "garpar_io",
    "mkss",
    "optimize",
    "StocksSet",
]
