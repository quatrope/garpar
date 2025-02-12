# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to dump and load stocks sets into hdf5.

This module provides functions to dump and load stocks sets into HDF5 files.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt
import json
import platform
import sys

import h5py

import numpy as np

import pandas as pd

from .constants import GARPAR_METADATA_KEY, VERSION
from .core import StocksSet

# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_HDF5_METADATA = {
    "garpar": VERSION,
    "author_email": "jbcabral@unc.edu.ar, nluczywo@unc.edu.ar",
    "affiliation": "FAMAF-UNC, FCE-UNC, QuatroPe",
    "url": "https://github.com/quatrope/garpar",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "Python": sys.version,
}

_WINDOW_SIZE_KEY = "window_size"


# =============================================================================
# UTILS
# =============================================================================


def _df_to_sarray(df):
    """Convert a pandas DataFrame object to a numpy structured array.

    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    Parameters
    ----------
    df : DataFrame
        The data frame to convert

    Returns
    -------
    numpy.ndarray
        A numpy structured array representation of `df`
    """
    values = df.values
    cols = df.columns
    types = [(cols[i], df[k].dtype.type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    s_arr = np.zeros(values.shape[0], dtype)
    for i, k in enumerate(s_arr.dtype.names):
        s_arr[k] = values[:, i]
    return s_arr


# =============================================================================
# HDF 5
# =============================================================================


def to_hdf5(path_or_stream, ss, group="stocks set", **kwargs):
    """HDF5 file writer.

    It is responsible for storing a stocks set in HDF5 format.

    Parameters
    ----------
    path_or_stream : str or file-like.
        Path or file like objet to the h5 to store the stocks set.
    ss : garpar.stocks_set.StocksSet
        The stocks set to store.
    group : str (default="stocks set")
        The name of the group where the stocks set will be stored.
    kwargs :
        Extra arguments to the function ``h5py.File.create_dataset``.
    """
    from . import __version__

    # prepare metadata
    h5_metadata = _DEFAULT_HDF5_METADATA.copy()
    h5_metadata["garpar"] = __version__
    h5_metadata["utc_timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat()

    # prepare kwargs
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)

    with h5py.File(path_or_stream, "a") as fp:
        # the data
        prices = _df_to_sarray(ss.as_prices())
        weights = ss.weights.to_numpy()
        entropy = ss.entropy.to_numpy()
        grp_attrs = {
            _WINDOW_SIZE_KEY: ss.window_size,
            GARPAR_METADATA_KEY: json.dumps(ss.metadata.to_dict()),
        }

        # store
        grp = fp.create_group(group)
        grp.create_dataset(f"{group}_prices", data=prices, **kwargs)
        grp.create_dataset(f"{group}_weights", data=weights, **kwargs)
        grp.create_dataset(f"{group}_entropy", data=entropy, **kwargs)
        grp.attrs.update(grp_attrs)

        # h5 metadata
        fp.attrs.update(h5_metadata)


def read_hdf5(path_or_stream, group="stocks set"):
    """HDF5 file reader.

    Parameters
    ----------
    path_to_stream : String
        Path to hdf5 stream
    group : String
        The value to look inside hdf5 file

    Returns
    -------
    ss: garpar.core.stocks_set.StocksSet
        A StocksSet instance
    """
    with h5py.File(path_or_stream, "r") as fp:
        grp = fp[group]

        prices_ds = grp[f"{group}_prices"]
        prices = pd.DataFrame(prices_ds[:])

        weights_ds = grp[f"{group}_weights"]
        weights = weights_ds[:]

        entropy_ds = grp[f"{group}_entropy"]
        entropy = entropy_ds[:]

        window_size = grp.attrs[_WINDOW_SIZE_KEY]
        metadata = json.loads(grp.attrs[GARPAR_METADATA_KEY])

    ss = StocksSet(
        prices,
        weights=weights,
        entropy=entropy,
        window_size=window_size,
        metadata=metadata,
    )

    return ss
