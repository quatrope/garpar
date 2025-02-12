# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================
"""Constants file."""

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # Python < 3.10
    import importlib_metadata


# =============================================================================
# CONSTANTS
# =============================================================================

#: Version of the package
VERSION = importlib_metadata.version("garpar")

#: Tolerance value for numerical operations
EPSILON = 1e-9


#: Key used to store metadata in a :class:`garpar.StocksSet`
GARPAR_METADATA_KEY = "__garpar_metadata__"
