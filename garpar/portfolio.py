# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import attr
from attr import validators as vldt

import pandas as pd

from .plot import PortfolioPlotter

GARPAR_METADATA_KEY = "__garpar_metadata__"


@attr.s(slots=True, frozen=True, cmp=False)
class Metadata:
    initial_prices = attr.ib(validator=vldt.instance_of(pd.Series))
    entropy = attr.ib(validator=vldt.optional(vldt.instance_of(float)))
    window_size = attr.ib(validator=vldt.optional(vldt.instance_of(int)))

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.initial_prices.equals(other.initial_prices)
            and self.entropy == other.entropy
            and self.window_size == other.window_size
        )

    def copy(self):
        return Metadata(
            initial_prices=self.initial_prices.copy(True),
            entropy=self.entropy,
            window_size=self.window_size,
        )


@attr.s(repr=False, cmp=False)
class Portfolio:

    _df = attr.ib(validator=vldt.instance_of(pd.DataFrame))

    def __attrs_post_init__(self):
        metadata = self._df.attrs[GARPAR_METADATA_KEY]
        if not isinstance(metadata, Metadata):
            raise TypeError(
                f"{GARPAR_METADATA_KEY} metadata must be an instance of "
                "'garpar.portfolio.Metadata'"
            )

    # ALTERNATIVE CONSTRUCTOR
    @classmethod
    def from_dfkws(cls, df, **kwargs):
        dfwmd = df.copy()
        dfwmd.attrs[GARPAR_METADATA_KEY] = Metadata(**kwargs)
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

    def __getattr__(self, a):
        metadata = attr.asdict(self._df.attrs[GARPAR_METADATA_KEY])
        if a in metadata:
            return metadata[a]
        return getattr(self._df, a)

    def __dir__(self):
        metadata_dir = list(attr.fields_dict(Metadata))
        return super().__dir__() + dir(self._df) + metadata_dir

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

    # REPR ====================================================================

    def __repr__(self):
        kwargs = {"show_dimensions": False}

        # retrieve the original string
        original_string = self._df.to_string(**kwargs)

        days, cols = self.shape
        dim = f"{days} days x {cols} stocks"

        # add dimension
        string = f"{original_string}\nPortfolio [{dim}]"

        return string
