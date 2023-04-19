# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


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
from ..utils import df_temporal_header, Bunch, scalers, entropy_calculators

# =============================================================================
# CONSTANTS
# =============================================================================

GARPAR_METADATA_KEY = "__garpar_metadata__"


def _as_float_array(arr):
    return np.asarray(arr, dtype=float)


# =============================================================================
# PORTFOLIO
# =============================================================================
@attr.s(repr=False, cmp=False)
class Portfolio:
    _df = attr.ib(validator=vldt.instance_of(pd.DataFrame))
    _weights = attr.ib(converter=_as_float_array)
    _entropy = attr.ib(converter=_as_float_array)
    _window_size = attr.ib(
        converter=lambda v: (None if pd.isna(v) else int(v))
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
        stocks_number = self.stocks_number

        if (
            len(self._weights) != stocks_number
            or len(self._entropy) != stocks_number
        ):
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

        cols = len(prices.columns)

        if weights is None or not hasattr(weights, "__iter__"):
            weights = 1.0 if weights is None else weights
            weights = np.full(cols, weights)

        if entropy is None or not hasattr(entropy, "__iter__"):
            entropy = np.nan if entropy is None else entropy
            entropy = np.full(cols, entropy)

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
            and np.allclose(self._weights, other._weights, equal_nan=True)
            and np.allclose(self._entropy, other._entropy, equal_nan=True)
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

    # PROPERTIES ==============================================================
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
    def delisted(self):
        dlstd = (self._df == 0.0).any(axis="rows")
        dlstd.name = "Delisted"
        return dlstd

    @property
    def shape(self):
        return self._df.shape

    # UTILS ===================================================================
    def copy(
        self,
        df=None,
        weights=None,
        entropy=None,
        window_size=None,
        preserve_old_metadata=True,
        **metadata,
    ):
        new_prices_df = (self._df if df is None else df).copy(deep=True)
        new_weights = (self._weights if weights is None else weights).copy()
        new_entropy = (self._entropy if entropy is None else entropy).copy()
        new_window_size = (
            self._window_size if window_size is None else window_size
        )

        new_metadata = (
            self._metadata.to_dict() if preserve_old_metadata else {}
        )
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
        window_size_df = pd.Series(window_size, index=self.stocks).to_frame().T
        window_size_df.index = ["WSize"]

        # adding the metadata to the dataframe
        metadata = self._metadata.to_dict()

        # creamos el df destino
        df = pd.concat([weights_df, entropy_df, window_size_df, price_df])
        df.attrs.update({GARPAR_METADATA_KEY: metadata})

        return df

    def as_returns(self, **kwargs):
        return pypfopt.expected_returns.returns_from_prices(
            prices=self._df, **kwargs
        )

    def as_prices(self):
        return self._df.copy()

    # PRUNNING ================================================================

    def weights_prune(self, threshold=1e-4):
        """Corta el portfolio en un umbral de pesos."""

        # get all data to prune
        prices = self.as_prices()
        weights = self.weights
        entropy = self.entropy
        window_size = self.window_size
        metadata = self.metadata.to_dict()

        # which criteria we want to preserve
        mask = weights[weights >= threshold].index

        # prune!
        pruned_prices = prices[weights[mask].index]
        pruned_weights = weights[mask].to_numpy()
        pruned_entropy = entropy[mask].to_numpy()

        # pruned pf
        pruned_pf = self.from_dfkws(
            pruned_prices,
            weights=pruned_weights,
            entropy=pruned_entropy,
            window_size=window_size,
            **metadata,
        )

        return pruned_pf

    wprune = weights_prune

    def delisted_prune(self):
        # get all data to prune
        prices = self.as_prices()
        weights = self.weights
        entropy = self.entropy
        metadata = self.metadata.to_dict()
        window_size = self.window_size

        # mask of not delisted
        dlstd = self.delisted
        mask = dlstd.index[~dlstd]

        # prune!
        pruned_prices = prices[mask].copy()
        pruned_weights = weights[mask].to_numpy()
        pruned_entropy = entropy[mask].to_numpy()

        # pruned pf
        pruned_pf = self.from_dfkws(
            pruned_prices,
            weights=pruned_weights,
            entropy=pruned_entropy,
            window_size=window_size,
            **metadata,
        )

        return pruned_pf

    dprune = delisted_prune

    # SCALE WEIGHTS ===========================================================

    _SCALERS = {
        "proportion": scalers.proportion_scaler,
        "minmax": scalers.minmax_scaler,
        "max": scalers.max_scaler,
        "std": scalers.standar_scaler,
    }

    def scale_weights(self, scaler="proportion"):
        """Reajusta los pesos en un rango de [0, 1]"""
        scaler = self._SCALERS.get(scaler, scaler)
        if not callable(scaler):
            saler_set = set(self._SCALERS)
            raise ValueError(
                f"'scaler' must be a one of '{saler_set}' or callable"
            )

        scaled_weights = scaler(self.weights.to_numpy())
        return self.copy(weights=scaled_weights)

    # CALCULATE ENTROPY =======================================================

    _ENTROPY_CALCULATORS = {"shannon": entropy_calculators.shannon}

    def refresh_entropy(self, entropy="shannon", entropy_kws=None):
        entropy_calc = self._ENTROPY_CALCULATORS.get(entropy, entropy)
        if not callable(entropy_calc):
            entropy_calc_set = set(self._ENTROPY_CALCULATORS)
            raise ValueError(
                f"'entropy' must be a one of '{entropy_calc_set}' or callable"
            )

        entropy_kws = {} if entropy_kws is None else entropy_kws

        new_entropy = entropy_calc(
            self.as_prices(), window_size=self.window_size, **entropy_kws
        )

        return self.copy(entropy=new_entropy)

    # REPR ====================================================================

    def _pd_fmt_serie(self, serie):
        print(serie)
        arr = serie.to_numpy(na_value=np.nan)
        return pd_fmt.format_array(arr, None, na_rep="?")

    def _get_sw_headers(self):
        """Columns names with weights and entropy."""
        headers = []
        fmt_weights = self._pd_fmt_serie(self.weights)
        fmt_entropy = self._pd_fmt_serie(self.entropy)
        for c, w, h in zip(self._df.columns, fmt_weights, fmt_entropy):
            header = f"{c}[W{w}, H{h}]"
            headers.append(header)
        return headers

    def _get_dxs_dimensions(self):
        """Dimension foote with dxs (Days x Stock)."""
        (days, stocks), wsize = self.shape, self.window_size
        wsize = "?" if pd.isna(wsize) else wsize
        dim = f"{days} days x {stocks} stocks - W.Size {wsize}"
        return dim

    def __repr__(self):
        header = self._get_sw_headers()
        dimensions = self._get_dxs_dimensions()

        with df_temporal_header(self._df, header) as df:
            with pd.option_context("display.show_dimensions", False):
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
        with df_temporal_header(self._df, header) as df:
            with pd.option_context("display.show_dimensions", False):
                original_html = df._repr_html_()

        # add dimension
        html = (
            "<div class='portfolio'>\n"
            f"{original_html}"
            f"<em class='portfolio-dim'>{dimensions}</em>\n"
            "</div>"
        )

        return html
