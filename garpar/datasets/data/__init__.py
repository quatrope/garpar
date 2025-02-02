# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""MERVAL dataset."""

import os
import pathlib

import pandas as pd

from ...core import StocksSet

DATA_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


def load_MERVAL(imputation="ffill", first=None, last=None):
    """Argentine stock market prices (MERVAL)."""
    df = pd.read_csv(DATA_PATH / "merval.csv", index_col="Days")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # pd.to_datetime(None) -> None
    first, last = pd.to_datetime(first), pd.to_datetime(last)

    if imputation in ("backfill", "bfill"):
        df.bfill(inplace=True)
    elif imputation in ("pad", "ffill"):
        df.ffill(inplace=True)
    else:
        df.fillna(value=imputation, inplace=True)

    if first is not None:
        df = df[df.index >= first]
    if last is not None:
        df = df[df.index <= last]

    port = StocksSet.from_prices(
        df,
        weights=None,
        title="Merval",
        imputation=imputation,
        description=(
            "Argentine stock market prices (MERVAL). "
            "Unlisted shares were eliminated."
        ),
    )

    return port
