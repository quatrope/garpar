# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


"""Datasets module for Garpar project.

This module provides functions for market simulation. It is used to
generate random market data based on a distribution, entropy, and window size.
Additionally, it includes a function to load MERVAL data from January 2012 to
August 2022.

Key Features:
    - Market simulation
    - Data loading
    - Entropy-based simulation
    - MERVAL dataset

See Also
--------
    Wiston Adrián Risso,
    The informational efficiency and the financial crashes.
    https://doi.org/10.1016/j.ribaf.2008.02.005.
"""

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
