# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import contextlib

# =============================================================================
# FUNCTIONS
# =============================================================================


@contextlib.contextmanager
def df_temporal_header(df, header):
    original_header = df.columns
    try:
        df.columns = header
        df.columns.name = original_header.name
        yield df
    finally:
        df.columns = original_header
