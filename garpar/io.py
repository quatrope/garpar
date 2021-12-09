# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to dump and load portfolios into hdf5.

Based on: https://stackoverflow.com/a/30773118

"""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt
import platform
import sys

import h5py

import numpy as np

import pandas as pd

from . import __version__ as VERSION, portfolio

# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_METADATA = {
    "garpar": VERSION,
    "author_email": "nluczywo@unc.edu.ar",
    "affiliation": "FCE-UNC",
    "url": "https://github.com/quatrope/garpart",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "Python": sys.version,
}


# =============================================================================
# UTILS
# =============================================================================


def _df_to_sarray(df):
    """Convert a pandas DataFrame object to a numpy structured array.

    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    :param df: the data frame to convert
    :return: a numpy structured array representation of df



    """

    v = df.values
    cols = df.columns
    types = [(cols[i], df[k].dtype.type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z


# =============================================================================
# HDF 5
# =============================================================================


def to_hdf5(path_or_stream, portfolio, dataset="portfolio", **kwargs):
    """HDF5 file writer.

    It is responsible for storing a portfolio in HDF5 format.

    Parameters
    ----------
    path_or_stream : str or file-like.
        Path or file like objet to the h5 to store the portfolio.
    portfolio : garpar.portfolio.PortFolio
        The portfolio to store.
    dataset : str (default="portfolio")
        The name of the dataset where the portfolio will be stored.
    kwargs :
        Extra arguments to the function ``h5py.File.create_dataset``.

    """
    sarray = _df_to_sarray(portfolio._df)

    # # prepare metadata
    h5_metadata = _DEFAULT_METADATA.copy()
    h5_metadata["utc_timestamp"] = dt.datetime.utcnow().isoformat()

    # prepare kwargs
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)

    with h5py.File(path_or_stream, "a") as h5:
        ds = h5.create_dataset(dataset, data=sarray, **kwargs)
        ds.attrs.update(portfolio.metadata)

        h5.attrs.update(h5_metadata)


def read_hdf5(path_or_stream, dataset="portfolio"):
    with h5py.File(path_or_stream, "r") as h5:
        ds = h5[dataset]
        df = pd.DataFrame(ds[:])
        return portfolio.Portfolio.from_dfkws(df, **ds.attrs)
