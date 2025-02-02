# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""Test garpar IO module."""

# =============================================================================
# IMPORTS
# =============================================================================

from io import BytesIO

from garpar import StocksSet, garpar_io

import pandas as pd

# =============================================================================
# TESTS
# =============================================================================


def test_StocksSet_to_hdf5_read_hdf5():
    """Test StocksSet to hdf5 and read hdf5."""
    ss = StocksSet.from_prices(
        prices=pd.DataFrame(
            {
                "stock0": [1, 2, 3, 4, 5],
                "stock1": [10, 20, 30, 40, 50],
            },
        ),
        weights=2,
        entropy=0.5,
        window_size=5,
    )

    buff = BytesIO()
    garpar_io.to_hdf5(buff, ss)
    buff.seek(0)
    result = garpar_io.read_hdf5(buff)

    assert ss == result
