# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

# from io import BytesIO

# import pandas as pd

# from garpar.io import read_hdf5, to_hdf5
# from garpar.core.portfolio import Portfolio


# =============================================================================
# TESTS
# =============================================================================


# def test_Portfolio_to_hdf5_read_hdf5():
#     pf = Portfolio.from_dfkws(
#         df=pd.DataFrame(
#             {
#                 "stock0": [1, 2, 3, 4, 5],
#                 "stock1": [10, 20, 30, 40, 50],
#             },
#         ),
#         entropy=0.5,
#         window_size=5,
#     )

#     buff = BytesIO()
#     to_hdf5(buff, pf)
#     buff.seek(0)
#     result = read_hdf5(buff)

#     assert pf == result
