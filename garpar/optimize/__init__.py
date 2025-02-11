# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Garpar optimize module.

This module provides optimizers for Garpar project. It includes mean-variance
optimizers. And interfaces for such optimizers.

Key Features:
    - Portfolio optimization
    - Mean-variance models
    - Markowitz model

"""

# =============================================================================
# IMPORTS
# =============================================================================

from . import mean_variance

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["mean_variance"]
