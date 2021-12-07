# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import attr
from attr import validators as vldt

import pandas as pd

GARPAR_METADATA_KEY = "__garpar_metadata__"


@attr.s(slots=True, frozen=True)
class Metadata:
    initial_prices = attr.ib(validator=vldt.instance_of(pd.Series))
    entropy = attr.ib(validator=vldt.instance_of(float))
    window_size = attr.ib(validator=vldt.instance_of(int))


@attr.s(repr=False)
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
    def mkportfolio(cls, df, **kwargs):
        dfwmd = df.copy()
        dfwmd.attrs[GARPAR_METADATA_KEY] = Metadata(**kwargs)
        return cls(df=dfwmd)

    # INTERNALS
    def __len__(self):
        return len(self._df)

    def __getattr__(self, a):
        metadata = attr.asdict(self._df.attrs[GARPAR_METADATA_KEY])
        if a in metadata:
            return metadata[a]
        return getattr(self._df, a)

    def __dir__(self):
        metadata_dir = list(attr.fields_dict(Metadata))
        return super().__dir__() + dir(self._df) + metadata_dir

    def __repr__(self):
        return ""
        raise NotImplementedError()

    def _repr_html_(self):
        raise NotImplementedError()
