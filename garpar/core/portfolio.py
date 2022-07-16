# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

from collections.abc import Mapping

import attr
from attr import validators as vldt
from cvxpy import utilities

import numpy as np

import pandas as pd
from pandas.io.formats import format as pd_fmt

import pyquery as pq

import pypfopt

from . import (
    covcorr_acc,
    ereturns_acc,
    plot_acc,
    prices_acc,
    risk_acc,
    utilities_acc,
    div_acc,
)


# =============================================================================
# CONSTANTS
# =============================================================================

GARPAR_METADATA_KEY = "__garpar_metadata__"


# =============================================================================
# METADATA
# =============================================================================


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
    _weights = attr.ib(converter=np.asarray)

    # accessors
    plot = attr.field(
        init=False,
        default=attr.Factory(plot_acc.PortfolioPlotter, takes_self=True),
    )

    prices = attr.field(
        init=False,
        default=attr.Factory(prices_acc.PricesAccessor, takes_self=True),
    )

    ereturns = attr.field(
        init=False,
        default=attr.Factory(
            ereturns_acc.ExpectedReturnsAccessor, takes_self=True
        ),
    )

    covariance = attr.field(
        init=False,
        default=attr.Factory(covcorr_acc.CovarianceAccessor, takes_self=True),
    )

    correlation = attr.field(
        init=False,
        default=attr.Factory(covcorr_acc.CorrelationAccessor, takes_self=True),
    )

    risk = attr.field(
        init=False,
        default=attr.Factory(risk_acc.RiskAccessor, takes_self=True),
    )

    utilities = attr.field(
        init=False,
        default=attr.Factory(utilities_acc.UtilitiesAccessor, takes_self=True),
    )

    div = attr.field(
        init=False,
        default=attr.Factory(div_acc.DiversificationAccessor, takes_self=True),
    )

    _VALID_METADATA = {
        "entropy": (float, np.floating),
        "window_size": (int, np.integer),
        "imputation": object,
        "description": str,
        "title": str,
    }

    def __attrs_post_init__(self):
        if len(self._weights) != len(self._df.columns):
            raise ValueError(
                f"The number of weights must be the same as number of stocks"
            )

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

        self._df.columns.name = "Stocks"
        self._df.index.name = "Days"

    # ALTERNATIVE CONSTRUCTOR
    @classmethod
    def from_dfkws(cls, df, weights=None, **kwargs):
        dfwmd = df.copy()
        dfwmd.attrs[GARPAR_METADATA_KEY] = Metadata(kwargs)

        if weights is None or not hasattr(weights, "__iter__"):
            cols = len(dfwmd.columns)
            weights = 1.0 / cols if weights is None else weights
            weights = np.full(cols, weights, dtype=float)

        return cls(df=dfwmd, weights=weights)

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

    def __getitem__(self, key):
        df = self._df.__getitem__(key)
        if isinstance(df, pd.Series):
            df = df.to_frame()

        weights = self.weights
        weights = weights[weights.index.isin(df.columns)].to_numpy()

        return Portfolio.from_dfkws(df=df, weights=weights, **self.metadata)

    # UTILS ===================================================================
    @property
    def weights(self):
        return pd.Series(self._weights, index=self._df.columns, name="Weights")

    @property
    def stocks(self):
        return self._df.columns.copy()

    @property
    def metadata(self):
        return self._df.attrs[GARPAR_METADATA_KEY]

    @property
    def shape(self):
        return self._df.shape

    def copy(self):
        copy_df = self._df.copy(deep=True)
        copy_weights = self._weights.copy()

        metadata = copy_df.attrs[GARPAR_METADATA_KEY].copy()
        copy_df.attrs[GARPAR_METADATA_KEY] = metadata

        return Portfolio(copy_df, weights=copy_weights)

    def to_hdf5(self, stream_or_buff, **kwargs):
        from .. import io

        return io.to_hdf5(stream_or_buff, self, **kwargs)

    def to_dataframe(self):
        df = self._df.copy(deep=True)

        # transform the weitgh "series" into a compatible dataframe
        weights_df = self.weights.to_frame().T
        weights_df.index = ["Weights"]

        # creating metadata rows in another df with the same columns of df
        metadata = df.attrs.pop(GARPAR_METADATA_KEY)
        mindex, mcols = sorted(metadata), {}
        for col in df.columns:
            mcols[col] = [metadata[mdi] for mdi in mindex]
        md_df = pd.DataFrame(mcols, index=mindex)

        return pd.concat([weights_df, md_df, df])

    def prune(self, threshold=0.0001):
        """Corta el portfolio en un umbral de pesos."""
        weights = self.weights

        mask = np.greater_equal(weights, threshold)

        pruned_df = self._df[weights[mask].index].copy()
        pruned_weights = weights[mask].to_numpy()

        return Portfolio(pruned_df, pruned_weights)

    def scale_weights(self):
        """Reajusta los pesos en un rango de [0, 1]"""
        scaled_weights = self._weights / self._weights.sum()
        return Portfolio(self._df.copy(), scaled_weights)

    def as_returns(self, **kwargs):
        return pypfopt.expected_returns.returns_from_prices(
            prices=self._df, **kwargs
        )

    # REPR ====================================================================

    def _get_sw_headers(self):
        """Columns names with weights."""
        headers = []
        fmt_weights = pd_fmt.format_array(self.weights, None)
        for c, w in zip(self._df.columns, fmt_weights):
            header = f"{c}[\u2696{w}]"
            headers.append(header)
        return headers

    def _get_dxs_dimensions(self):
        """Dimension foote with dxs (Days x Stock)."""
        days, cols = self.shape
        dim = f"{days} days x {cols} stocks"
        return dim

    def __repr__(self):
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        max_cols = pd.get_option("display.max_columns")
        max_colwidth = pd.get_option("display.max_colwidth")

        if pd.get_option("display.expand_frame_repr"):
            width, _ = pd.io.formats.console.get_console_size()
        else:
            width = None
        original_string = self._df.to_string(
            max_rows=max_rows,
            min_rows=min_rows,
            max_cols=max_cols,
            line_width=width,
            max_colwidth=max_colwidth,
            show_dimensions=False,
            header=self._get_sw_headers(),
        )

        dim = self._get_dxs_dimensions()

        # add dimension
        string = f"{original_string}\nPortfolio [{dim}]"

        return string

    def _repr_html_(self):
        """Return a html representation for a the Portfolio.

        Mainly for IPython notebook.
        """
        header = dict(zip(self._df.columns, self._get_sw_headers()))
        dimensions = self._get_dxs_dimensions()

        # retrieve the original string
        with pd.option_context("display.show_dimensions", False):
            original_html = self._df._repr_html_()

        # add dimension
        html = (
            "<div class='portfolio'>\n"
            f"{original_html}"
            f"<em class='portfolio-dim'>{dimensions}</em>\n"
            "</div>"
        )

        # now we need to change the table header
        d = pq.PyQuery(html)
        for th in d("div.portfolio table.dataframe > thead > tr > th"):
            stock = th.text
            th.text = header.get(stock, stock)

        return str(d)
