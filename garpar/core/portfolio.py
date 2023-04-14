# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

from collections.abc import Mapping

import attr
from attr import validators as vldt

import numpy as np

import pandas as pd
from pandas.io.formats import format as pd_fmt

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
from ..utils import df_temporal_header, Bunch

# =============================================================================
# CONSTANTS
# =============================================================================

GARPAR_METADATA_KEY = "__garpar_metadata__"


# =============================================================================
# PORTFOLIO
# =============================================================================
@attr.s(repr=False, cmp=False)
class Portfolio:
    _df = attr.ib(validator=vldt.instance_of(pd.DataFrame))
    _weights = attr.ib(converter=np.asarray)
    _entropy = attr.ib(converter=np.asarray)
    _window_size = attr.ib(
        converter=lambda v: (pd.NA if pd.isna(v) else int(v))
    )
    _metadata = attr.ib(factory=dict, converter=lambda d: Bunch("metadata", d))

    # accessors
    plot = attr.ib(
        init=False,
        default=attr.Factory(plot_acc.PortfolioPlotter, takes_self=True),
    )

    prices = attr.ib(
        init=False,
        default=attr.Factory(prices_acc.PricesAccessor, takes_self=True),
    )

    ereturns = attr.ib(
        init=False,
        default=attr.Factory(
            ereturns_acc.ExpectedReturnsAccessor, takes_self=True
        ),
    )

    covariance = attr.ib(
        init=False,
        default=attr.Factory(covcorr_acc.CovarianceAccessor, takes_self=True),
    )
    cov = covariance

    correlation = attr.ib(
        init=False,
        default=attr.Factory(covcorr_acc.CorrelationAccessor, takes_self=True),
    )

    corr = correlation

    risk = attr.ib(
        init=False,
        default=attr.Factory(risk_acc.RiskAccessor, takes_self=True),
    )

    utilities = attr.ib(
        init=False,
        default=attr.Factory(utilities_acc.UtilitiesAccessor, takes_self=True),
    )

    div = attr.ib(
        init=False,
        default=attr.Factory(div_acc.DiversificationAccessor, takes_self=True),
    )

    def __attrs_post_init__(self):
        if len(self._weights) != len(self._entropy) != len(self._df.columns):
            raise ValueError(
                "The number of weights and entropy must "
                "be the same as number of stocks"
            )

        self._df.columns.name = "Stocks"
        self._df.index.name = "Days"

    # ALTERNATIVE CONSTRUCTOR
    @classmethod
    def from_dfkws(
        cls, df, weights=None, entropy=None, window_size=None, **metadata
    ):
        prices = df.copy()

        if weights is None or not hasattr(weights, "__iter__"):
            cols = len(prices.columns)
            weights = 1.0 if weights is None else weights
            weights = np.full(cols, weights, dtype=float)

        if entropy is None or not hasattr(entropy, "__iter__"):
            cols = len(prices.columns)
            entropy = pd.NA if entropy is None else entropy
            entropy = np.full(cols, entropy, dtype=object)

        pf = cls(
            df=prices,
            weights=weights,
            entropy=entropy,
            window_size=window_size,
            metadata=metadata,
        )

        return pf

    # INTERNALS
    def __len__(self):
        return len(self._df)

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self._df.equals(other._df)
            and np.array_equal(self._weights, other._weights, equal_nan=True)
            and np.array_equal(self._entropy, other._entropy, equal_nan=True)
            and self._window_size == other._window_size
            and self._metadata == other._metadata
        )

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, key):
        df = self._df.__getitem__(key)
        if isinstance(df, pd.Series):
            df = df.to_frame()

        weights = self.weights
        weights = weights[weights.index.isin(df.columns)].to_numpy()

        entropy = self.entropy
        entropy = entropy[entropy.index.isin(df.columns)].to_numpy()

        window_size = self.window_size
        metadata = dict(self.metadata)

        sliced = Portfolio.from_dfkws(
            df=df,
            weights=weights,
            entropy=entropy,
            window_size=window_size,
            **metadata,
        )

        return sliced

    # UTILS ===================================================================
    @property
    def weights(self):
        return pd.Series(self._weights, index=self._df.columns, name="Weights")

    @property
    def entropy(self):
        return pd.Series(self._entropy, index=self._df.columns, name="Entropy")

    @property
    def stocks(self):
        return self._df.columns.to_numpy()

    @property
    def stocks_number(self):
        return len(self._df.columns)

    @property
    def metadata(self):
        return self._metadata

    @property
    def window_size(self):
        return self._window_size

    @property
    def shape(self):
        return self._df.shape

    def copy(
        self, df=None, weights=None, entropy=None, window_size=None, **metadata
    ):
        new_prices_df = (self._df if df is None else df).copy(deep=True)
        new_weights = (self._weights if weights is None else weights).copy()
        new_entropy = (self._entropy if entropy is None else entropy).copy()
        new_window_size = (
            self._window_size if window_size is None else window_size
        )

        new_metadata = self._metadata.to_dict()
        new_metadata.update(metadata)

        new_pf = Portfolio(
            new_prices_df,
            weights=new_weights,
            entropy=new_entropy,
            window_size=new_window_size,
            metadata=new_metadata,
        )

        return new_pf

    def to_hdf5(self, stream_or_buff, **kwargs):
        from .. import io

        return io.to_hdf5(stream_or_buff, self, **kwargs)

    def to_dataframe(self):
        price_df = self._df.copy(deep=True)

        # transform the weighs "series" into a compatible dataframe
        weights_df = self.weights.to_frame().T
        weights_df.index = ["Weights"]

        # The same for entropy
        entropy_df = self.entropy.to_frame().T
        entropy_df.index = ["Entropy"]

        # window size
        window_size = np.full(self.stocks_number, self.window_size)
        window_size_df = pd.Series(window_size).to_frame().T
        window_size_df.index = ["WSize"]

        # adding the metadata to the dataframe
        metadata = self._metadata.to_dict()

        # creamos el df destino
        df = pd.concat([weights_df, entropy_df, window_size_df, price_df])
        df.attrs.update({GARPAR_METADATA_KEY: metadata})

        return df

    @property
    def delisted(self):
        dlstd = (self._df == 0.0).any(axis="rows")
        dlstd.name = "Delisted"
        return dlstd

    # def weights_prune(self, threshold=1e-4):
    #     """Corta el portfolio en un umbral de pesos."""
    #     weights = self.weights

    #     mask = np.greater_equal(weights, threshold)

    #     pruned_df = self._df[weights[mask].index].copy()
    #     pruned_weights = weights[mask].to_numpy()

    #     return Portfolio(pruned_df, pruned_weights)

    # wprune = weights_prune

    # def delisted_prune(self):
    #     dlstd = self.delisted

    #     not_delisted = dlstd.index[~dlstd]

    #     pruned_df = self._df[not_delisted].copy()
    #     pruned_weights = self.weights[not_delisted].to_numpy()

    #     return Portfolio(pruned_df, pruned_weights)

    # dprune = delisted_prune

    # def proportional_weights(self):
    #     """Reajusta los pesos en un rango de [0, 1]"""
    #     scaled_weights = self._weights / self._weights.sum()
    #     return self.copy(weights=scaled_weights)

    def as_returns(self, **kwargs):
        return pypfopt.expected_returns.returns_from_prices(
            prices=self._df, **kwargs
        )

    def as_prices(self):
        return self._df.copy()

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
        header = self._get_sw_headers()
        dimensions = self._get_dxs_dimensions()

        with (
            df_temporal_header(self._df, header) as df,
            pd.option_context("display.show_dimensions", False),
        ):
            original_string = repr(df)

        # add dimension
        string = f"{original_string}\nPortfolio [{dimensions}]"

        return string

    def _repr_html_(self):
        """Return a html representation for a the Portfolio.

        Mainly for IPython notebook.
        """
        header = self._get_sw_headers()
        dimensions = self._get_dxs_dimensions()

        # retrieve the original string
        with (
            df_temporal_header(self._df, header) as df,
            pd.option_context("display.show_dimensions", False),
        ):
            original_html = df._repr_html_()

        # add dimension
        html = (
            "<div class='portfolio'>\n"
            f"{original_html}"
            f"<em class='portfolio-dim'>{dimensions}</em>\n"
            "</div>"
        )

        return html
