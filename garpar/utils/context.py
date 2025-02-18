# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Context util for Garpar project."""


# =============================================================================
# IMPORTS
# =============================================================================

import contextlib


# =============================================================================
# TEMPORAL HEADER
# =============================================================================


@contextlib.contextmanager
def df_temporal_header(df, header, name=None):
    """Temporarily replaces a DataFrame columns names.

    Optionally also assign another name to the columns.

    Parameters
    ----------
    header : sequence
        The new names of the columns.
    name : str or None (default None)
        New name for the index containing the columns in the DataFrame. If
        'None' the original name of the columns present in the DataFrame is
        preserved.

    """
    original_header = df.columns
    original_name = original_header.name

    name = original_name if name is None else name
    try:
        df.columns = header
        df.columns.name = name
        yield df
    finally:
        df.columns = original_header
        df.columns.name = original_name
