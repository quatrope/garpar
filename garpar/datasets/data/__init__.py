import os
import pathlib

import pandas as pd

from ...core import Portfolio

DATA_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


def load_merval2021_2022(imputation="ffill"):
    """Argentine stock market prices (MERVAL) from 2020 to early 2022. \
    Unlisted shares were eliminated.

    """
    df = pd.read_csv(DATA_PATH / "merval2020-2022.csv", index_col="Days")
    df.index = pd.to_datetime(df.index)

    if imputation in ("backfill", "bfill", "pad", "ffill"):
        df.fillna(method=imputation, inplace=True)
    else:
        df.fillna(value=imputation, inplace=True)

    port = Portfolio.from_dfkws(
        df,
        weights=None,
        title="Merval 2020-2022",
        imputation=imputation,
        description=(
            "Argentine stock market prices (MERVAL) from 2020 to early 2022. "
            "Unlisted shares were eliminated."
        ),
    )

    return port
