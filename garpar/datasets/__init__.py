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

"""Datasets subpackage of Garpar project.

This package provides everything related to StocksSets generation. It includes
StocksSet makers for simulation of market data. It includes a base class for
StocksSet makers and a base class for random entropy-based functions for market
simulation. It also includes a base class for multi-sector StocksSet makers.
Additionally, it includes a function to create a StocksSet from MERVAL data
from January 2012 to August 2022.

Key Features:
    - Entropy-based market simulation
    - Multi-sector StocksSet creation
    - MERVAL dataset

See Also
--------
    Wiston Adrián Risso,
    The informational efficiency and the financial crashes.
    https://doi.org/10.1016/j.ribaf.2008.02.005.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .data import load_MERVAL
from .ds_base import RandomEntropyStocksSetMakerABC, StocksSetMakerABC
from .multisector import MultiSector, make_multisector
from .risso import (
    RissoLevyStable,
    RissoMixin,
    RissoNormal,
    RissoUniform,
    make_risso_levy_stable,
    make_risso_normal,
    make_risso_uniform,
)

# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "StocksSetMakerABC",
    "RandomEntropyStocksSetMakerABC",
    "MultiSector",
    "RissoLevyStable",
    "RissoNormal",
    "RissoUniform",
    "RissoMixin",
    "make_multisector",
    "make_risso_normal",
    "make_risso_levy_stable",
    "make_risso_uniform",
    "load_MERVAL",
]
