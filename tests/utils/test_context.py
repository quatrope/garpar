# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE
# =============================================================================
# DOCS
# =============================================================================

"""Test for garpar.utils.context"""


# =============================================================================
# IMPORTS
# =============================================================================

from garpar.utils import context

import pandas as pd


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_df_temporal_header():
    df = pd.DataFrame({"x": [1], "y": [2]})
    df.columns.name = "original"

    with context.df_temporal_header(df, ["a", "b"], "replaced") as df:
        pd.testing.assert_index_equal(
            df.columns, pd.Index(["a", "b"], name="replaced")
        )

    pd.testing.assert_index_equal(
        df.columns, pd.Index(["x", "y"], name="original")
    )
