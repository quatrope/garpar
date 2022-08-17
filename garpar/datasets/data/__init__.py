import datetime as dt
import os
import pathlib

import dateutil.parser as dtparser

import pandas as pd

from ...core import Portfolio

DATA_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


def _as_date(date):
    return pd.to_datetime(
        date
        if isinstance(date, (dt.datetime, dt.date))
        else dtparser.parse(date)
    )


def load_MERVAL(imputation="ffill", first=None, last=None):
    """Argentine stock market prices (MERVAL). \
    Unlisted shares were eliminated.

    """
    df = pd.read_csv(DATA_PATH / "merval.csv", index_col="Days")
    df.index = pd.to_datetime(df.index)

    last = dt.date.today() if last is None else last
    first = last - dt.timedelta(weeks=52) if first is None else first

    first_parsed = pd.to_datetime(first)
    last_parsed = pd.to_datetime(last)

    if imputation in ("backfill", "bfill", "pad", "ffill"):
        df.fillna(method=imputation, inplace=True)
    else:
        df.fillna(value=imputation, inplace=True)

    filter_df = df[(df.index >= first_parsed) & (df.index <= last_parsed)]
    filter_df.sort_index(inplace=True)

    del df

    port = Portfolio.from_dfkws(
        filter_df,
        weights=None,
        title="Merval",
        imputation=imputation,
        description=(
            "Argentine stock market prices (MERVAL). "
            "Unlisted shares were eliminated."
        ),
    )

    return port
