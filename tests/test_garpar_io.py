# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


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
    ss = StocksSet.from_dfkws(
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
