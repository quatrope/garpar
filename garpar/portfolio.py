# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

from collections.abc import Mapping

import attr
from attr import validators as vldt

import pandas as pd

from .plot import PortfolioPlotter

# =============================================================================
# CONSTANTS
# =============================================================================

GARPAR_METADATA_KEY = "__garpar_metadata__"


# =============================================================================
# UTILS
# =============================================================================
@attr.s(cmp=False, frozen=True, slots=True)
class _WrapSlicer:

    slicer = attr.ib()
    frame_to = attr.ib()

    def __getitem__(self, slice):
        result = self.slicer.__getitem__(slice)
        if isinstance(result, pd.DataFrame):
            result = self.frame_to(result)
        return result


@attr.s(slots=True, frozen=True, repr=False)
class Metadata(Mapping):

    _data = attr.ib(validator=vldt.instance_of(Mapping))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, k):
        return self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __dir__(self):
        return super().__dir__() + list(self._data)

    def __repr__(self):
        content = ", ".join(self._data)
        return f"metadata({content})"

    def __getattr__(self, a):
        try:
            return self[a]
        except KeyError:
            raise AttributeError(a)

    def copy(self):
        return Metadata(data=self._data.copy())


# =============================================================================
# PORTFOLIO
# =============================================================================
@attr.s(repr=False, cmp=False)
class Portfolio:

    _df = attr.ib(validator=vldt.instance_of(pd.DataFrame))

    _DF_WHITELIST = ["info", "describe", "cumsum", "cumprod", "T", "transpose"]
    _VALID_METADATA = {"entropy": float, "window_size": int}

    def __attrs_post_init__(self):
        metadata = self._df.attrs[GARPAR_METADATA_KEY]
        if not isinstance(metadata, Metadata):
            raise TypeError(
                f"{GARPAR_METADATA_KEY} metadata must be an instance of "
                "'garpar.portfolio.Metadata'"
            )
        for k, v in metadata.items():
            if k not in self._VALID_METADATA:
                raise ValueError(f"Invalid metadata '{k}'")
            mtype = self._VALID_METADATA[k]
            if not isinstance(v, mtype):
                raise TypeError(
                    f"Metadata '{k}' must be instance of {mtype}. "
                    f"Found {type(v)}"
                )

    # ALTERNATIVE CONSTRUCTOR
    @classmethod
    def from_dfkws(cls, df, **kwargs):
        dfwmd = df.copy()
        dfwmd.attrs[GARPAR_METADATA_KEY] = Metadata(kwargs)
        return cls(df=dfwmd)

    # INTERNALS
    def __len__(self):
        return len(self._df)

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self._df.equals(other._df)
            and self._df.attrs[GARPAR_METADATA_KEY]
            == other._df.attrs[GARPAR_METADATA_KEY]
        )

    def __ne__(self, other):
        return not self == other

    # GETATTR =================================================================

    def __getattr__(self, a):
        if a not in dir(self):
            raise AttributeError(a)
        metadata = self._df.attrs[GARPAR_METADATA_KEY]
        if a in metadata:
            return metadata[a]
        return getattr(self._df, a)

    def __dir__(self):
        metadata_dir = list(self._df.attrs[GARPAR_METADATA_KEY])
        df_dir = [d for d in dir(self._df) if d in self._DF_WHITELIST]
        return super().__dir__() + df_dir + metadata_dir

    # UTILS ===================================================================
    @property
    def shape(self):
        return self._df.shape

    @property
    def plot(self):
        return PortfolioPlotter(self)

    def copy(self):
        copy_df = self._df.copy(deep=True)

        metadata = copy_df.attrs[GARPAR_METADATA_KEY].copy()
        copy_df.attrs[GARPAR_METADATA_KEY] = metadata

        return Portfolio(copy_df)

    def to_hdf5(self, stream_or_buff):
        pass

    def to_dataframe(self):
        df = self._df.copy(deep=True)

        # creating metadata rows in another df with the same columns of df
        metadata = df.attrs.pop(GARPAR_METADATA_KEY)
        mindex, mcols = sorted(metadata), {}
        for col in df.columns:
            mcols[col] = [metadata[mdi] for mdi in mindex]
        md_df = pd.DataFrame(mcols, index=mindex)

        return pd.concat([md_df, df])

    # SLICERS =================================================================

    def __getitem__(self, slice):
        return _WrapSlicer(self._df, frame_to=type(self)).__getitem__(slice)

    @property
    def iloc(self):
        return _WrapSlicer(self._df.iloc, frame_to=type(self))

    @property
    def loc(self):
        return _WrapSlicer(self._df.loc, frame_to=type(self))

    # REPR ====================================================================

    def __repr__(self):
        with pd.option_context("display.show_dimensions", False):
            original_string = repr(self._df)

        days, cols = self.shape
        dim = f"{days} days x {cols} stocks"

        # add dimension
        string = f"{original_string}\nPortfolio [{dim}]"

        return string
