# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Optimize subpackage of Garpar project.

This subpackage offers foundational classes for developing mean-variance 
optimization models tailored for StocksSets, featuring various implementations 
leveraging PyPortfolioOpt.

Key Features:
    - Portfolio optimization
    - Mean-variance models
    - Markowitz model

See Also
--------
    PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/

"""

# =============================================================================
# IMPORTS
# =============================================================================

from . import mean_variance

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["mean_variance"]
