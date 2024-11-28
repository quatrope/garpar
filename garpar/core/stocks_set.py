# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

"""StocksSet."""

import attr
from attr import validators as vldt

import numpy as np

import pandas as pd
from pandas.io.formats import format as pd_fmt

import pypfopt

from . import (
    covcorr_acc,
    div_acc,
    ereturns_acc,
    plot_acc,
    prices_acc,
    risk_acc,
    utilities_acc,
)
from ..utils import Bunch, df_temporal_header, entropy, scalers

# =============================================================================
# CONSTANTS
# =============================================================================

GARPAR_METADATA_KEY = "__garpar_metadata__"

_SCALERS = {
    "proportion": scalers.proportion_scaler,
    "minmax": scalers.minmax_scaler,
    "max": scalers.max_scaler,
    "std": scalers.standar_scaler,
}

_ENTROPY_CALCULATORS = {"shannon": entropy.shannon}


def _as_float_array(arr):
    return np.asarray(arr, dtype=float)


# =============================================================================
# STOCKS SET
# =============================================================================
@attr.s(repr=False, cmp=False)
class StocksSet:
    """
    Represents a financial stocks set with utilities for analysis and manipulation.
    """

    _prices_df = attr.ib(validator=vldt.instance_of(pd.DataFrame))
    _weights = attr.ib(converter=_as_float_array)
    _entropy = attr.ib(converter=_as_float_array)
    _window_size = attr.ib(converter=lambda v: (None if pd.isna(v) else int(v)))
    _metadata = attr.ib(factory=dict, converter=lambda d: Bunch("metadata", d))

    # accessors
    plot = attr.ib(
        init=False,
        default=attr.Factory(
            plot_acc.StocksSetPlotterAccessor, takes_self=True
        ),
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

    diversification = attr.ib(
        init=False,
        default=attr.Factory(
            div_acc.DiversificationMetricsAccessor, takes_self=True
        ),
    )
    div = diversification

    def __attrs_post_init__(self):
        """Initialize additional attributes and performs validation.

        Raises
        ------
        ValueError
            If the number of weights or entropy values does not match the number of stocks.
        """
        stocks_number = self.stocks_number

        if (
            len(self._weights) != stocks_number
            or len(self._entropy) != stocks_number
        ):
            raise ValueError(
                "The number of weights and entropy must "
                "be the same as number of stocks"
            )

        self._prices_df.columns.name = "Stocks"
        self._prices_df.index.name = "Days"

    # ALTERNATIVE CONSTRUCTOR
    @classmethod
    def from_dfkws(
        cls,
        prices,
        weights=None,
        entropy=None,
        window_size=None,
        stocks=None,
        **metadata,
    ):
        """Alternative constructor to create a StocksSet instance from various inputs.

        Parameters
        ----------
        prices : pd.DataFrame or array-like
            DataFrame or array-like object containing the prices of the assets.
        weights : array-like, optional
            Array of asset weights in the stocks set.
        entropy : array-like, optional
            Array of entropy values associated with the assets.
        window_size : int or None, optional
            Window size for rolling calculations, if applicable.
        stocks : array-like, optional
            List of stock names.
        **metadata
            Additional metadata related to the stocks set.

        Returns
        -------
        StocksSet
            A new StocksSet instance.
        """
        prices = (
            prices.copy()
            if isinstance(prices, pd.DataFrame)
            else pd.DataFrame(prices)
        )

        stocks_number = len(prices.columns)

        if weights is None or not hasattr(weights, "__iter__"):
            weights = 1.0 if weights is None else weights
            weights = np.full(stocks_number, weights)

        if entropy is None or not hasattr(entropy, "__iter__"):
            entropy = np.nan if entropy is None else entropy
            entropy = np.full(stocks_number, entropy)

        if stocks is not None:
            prices.columns = stocks

        ss = cls(
            prices_df=prices,
            weights=weights,
            entropy=entropy,
            window_size=window_size,
            metadata=metadata,
        )

        return ss

    # INTERNALS
    def __len__(self):
        """Return the number of days in the price DataFrame.

        Returns
        -------
        int
            Number of days in the price DataFrame.
        """
        return len(self._prices_df)

    def __eq__(self, other):
        """Check equality with another StocksSet instance.

        Parameters
        ----------
        other : StocksSet
            Another StocksSet instance to compare with.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        return (
            isinstance(other, type(self))
            and self._prices_df.equals(other._prices_df)
            and np.allclose(self._weights, other._weights, equal_nan=True)
            and np.allclose(self._entropy, other._entropy, equal_nan=True)
            and self._window_size == other._window_size
            and self._metadata == other._metadata
        )

    def __ne__(self, other):
        """Check inequality with another StocksSet instance.

        Parameters
        ----------
        other : StocksSet
            Another StocksSet instance to compare with.

        Returns
        -------
        bool
            True if not equal, False otherwise.
        """
        return not self == other

    def __getitem__(self, key):
        """Slices the StocksSet by the given key.

        Parameters
        ----------
        key : int, slice, or array-like
            The key to use for slicing.

        Returns
        -------
        StocksSet
            A new StocksSet instance sliced by the key.
        """
        prices = self._prices_df.__getitem__(key)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        weights = self.weights
        weights = weights[weights.index.isin(prices.columns)].to_numpy()

        entropy = self.entropy
        entropy = entropy[entropy.index.isin(prices.columns)].to_numpy()

        window_size = self.window_size
        metadata = dict(self.metadata)

        cls = type(self)
        sliced = cls(
            prices_df=prices,
            weights=weights,
            entropy=entropy,
            window_size=window_size,
            metadata=metadata,
        )

        return sliced

    # PROPERTIES ==============================================================
    @property
    def weights(self):
        """Return the weights as a pandas Series.

        Returns
        -------
        pd.Series
            Series of asset weights.
        """
        return pd.Series(
            self._weights, index=self._prices_df.columns, name="Weights"
        )

    @property
    def entropy(self):
        """Return the entropy values as a pandas Series.

        Returns
        -------
        pd.Series
            Series of entropy values.
        """
        return pd.Series(
            self._entropy, index=self._prices_df.columns, name="Entropy"
        )

    @property
    def stocks(self):
        """Return the stocks in the stocks set.

        Returns
        -------
        np.ndarray
            Array of stock names.
        """
        return self._prices_df.columns.to_numpy()

    @property
    def stocks_number(self):
        """Return the number of stocks in the stocks set.

        Returns
        -------
        int
            Number of stocks in the stocks set.
        """
        return len(self._prices_df.columns)

    @property
    def metadata(self):
        """Return the metadata as a Bunch object.

        Returns
        -------
        Bunch
            Bunch object containing metadata.
        """
        return self._metadata

    @property
    def window_size(self):
        """Return the window size for rolling calculations.

        Returns
        -------
        int or None
            Window size for rolling calculations.
        """
        return self._window_size

    @property
    def delisted(self):
        """Return a Series indicating if a stock has been delisted.

        Returns
        -------
        pd.Series
            Series with boolean values indicating delisted status.
        """
        dlstd = (self._prices_df == 0.0).any(axis="rows")
        dlstd.name = "Delisted"
        return dlstd

    @property
    def shape(self):
        """Return the shape of the price DataFrame.

        Returns
        -------
        tuple
            Shape of the price DataFrame.
        """
        return self._prices_df.shape

    # UTILS ===================================================================
    # TODO: Hacer que copy pueda elegir desde que dia hasta que dia copiar
    def copy(
        self,
        *,
        prices=None,
        weights=None,
        entropy=None,
        window_size=None,
        stocks=None,
        preserve_old_metadata=True,
        **metadata,
    ):
        """Create a copy of the StocksSet.

        Parameters
        ----------
        prices : pd.DataFrame, optional
            New prices DataFrame.
        weights : array-like, optional
            New weights array.
        entropy : array-like, optional
            New entropy array.
        window_size : int or None, optional
            New window size for rolling calculations.
        stocks : array-like, optional
            New list of stock names.
        preserve_old_metadata : bool, optional
            Whether to preserve old metadata.
        **metadata
            Additional metadata to include.

        Returns
        -------
        StocksSet
            A new StocksSet instance with the specified modifications.
        """
        new_prices_df = (self._prices_df if prices is None else prices).copy()
        new_weights = (self._weights if weights is None else weights).copy()
        new_entropy = (self._entropy if entropy is None else entropy).copy()
        new_window_size = (
            self._window_size if window_size is None else window_size
        )

        new_metadata = self._metadata.to_dict() if preserve_old_metadata else {}
        new_metadata.update(metadata)

        new_ss = self.from_dfkws(
            new_prices_df,
            weights=new_weights,
            entropy=new_entropy,
            window_size=new_window_size,
            stocks=stocks,
            **new_metadata,
        )

        return new_ss

    def to_hdf5(self, stream_or_buff, **kwargs):
        """Save the StocksSet to an HDF5 file.

        Parameters
        ----------
        stream_or_buff : str or file-like
            Path or file-like object to save the HDF5 file.
        **kwargs
            Additional arguments to pass to the HDF5 writer.

        Returns
        -------
        None
        """
        from .. import io

        return io.to_hdf5(stream_or_buff, self, **kwargs)

    def to_dataframe(self):
        """Convert the StocksSet object to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame representation of the StocksSet object.
        """
        price_df = self._prices_df.copy(deep=True)

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
        """Convert prices to returns using PyStocksSetOpt's expected_returns module.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to returns_from_prices function.

        Returns
        -------
        pd.DataFrame
            DataFrame of returns corresponding to the prices DataFrame.
        """
        return pypfopt.expected_returns.returns_from_prices(
            prices=self._prices_df, **kwargs
        )

    def as_prices(self):
        """Return a copy of the prices DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of the prices DataFrame.
        """
        return self._prices_df.copy()

    # PRUNNING ================================================================

    def weights_prune(self, threshold=1e-4):
        """Prune the stocks set based on a weight threshold.

        Parameters
        ----------
        threshold : float, optional
            Threshold below which weights are considered insignificant.

        Returns
        -------
        StocksSet
            A pruned StocksSet instance.
        """
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

        # pruned ss
        cls = type(self)

        pruned_ss = cls(
            prices_df=pruned_prices,
            weights=pruned_weights,
            entropy=pruned_entropy,
            window_size=window_size,
            metadata=metadata,
        )

        return pruned_ss

    wprune = weights_prune

    def delisted_prune(self):
        """Prunes the stocks set by removing delisted stocks.

        Returns
        -------
        StocksSet
            A pruned StocksSet instance.
        """
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

        # pruned ss
        cls = type(self)

        pruned_ss = cls(
            prices_df=pruned_prices,
            weights=pruned_weights,
            entropy=pruned_entropy,
            window_size=window_size,
            metadata=metadata,
        )

        return pruned_ss

    dprune = delisted_prune

    # SCALE WEIGHTS ===========================================================

    def scale_weights(self, *, scaler="proportion"):
        """Scales the weights to a specified range.

        Parameters
        ----------
        scaler : str or callable, optional
            Method or function to use for scaling weights.

        Returns
        -------
        StocksSet
            A StocksSet instance with scaled weights.
        """
        """Reajusta los pesos en un rango de [0, 1]"""
        scaler = _SCALERS.get(scaler, scaler)
        if not callable(scaler):
            saler_set = set(_SCALERS)
            raise ValueError(
                f"'scaler' must be a one of '{saler_set}' or callable"
            )

        scaled_weights = scaler(self.weights.to_numpy())
        return self.copy(weights=scaled_weights)

    # CALCULATE ENTROPY =======================================================
    def refresh_entropy(self, *, entropy="shannon", entropy_kws=None):
        """Refresh entropy values using a specified entropy calculation method.

        Parameters
        ----------
        entropy : str or callable, optional
            Method or function to use for calculating entropy.
        entropy_kws : dict, optional
            Additional keyword arguments for the entropy calculation function.

        Returns
        -------
        StocksSet
            A StocksSet instance with refreshed entropy values.
        """
        entropy_calc = _ENTROPY_CALCULATORS.get(entropy, entropy)
        if not callable(entropy_calc):
            entropy_calc_set = set(_ENTROPY_CALCULATORS)
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
        """Format a pandas Series for display.

        Parameters
        ----------
        serie : pd.Series
            Series to be formatted.

        Returns
        -------
        str
            Formatted string representation of the Series.
        """
        arr = serie.to_numpy(na_value=np.nan)
        return pd_fmt.format_array(arr, None, na_rep="?")

    def _get_sw_headers(self):
        """Return column headers for weights and entropy.

        Returns
        -------
        list of str
            List of formatted column headers.
        """
        headers = []
        fmt_weights = self._pd_fmt_serie(self.weights)
        fmt_entropy = self._pd_fmt_serie(self.entropy)
        for c, w, h in zip(self._prices_df.columns, fmt_weights, fmt_entropy):
            header = f"{c}[W{w}, H{h}]"
            headers.append(header)
        return headers

    def _get_dxs_dimensions(self):
        """Return dimensions information for the StocksSet.

        Returns
        -------
        str
            Dimension information formatted as string.
        """
        (days, stocks), wsize = self.shape, self.window_size
        wsize = "?" if pd.isna(wsize) else wsize
        dim = f"{days} days x {stocks} stocks - W.Size {wsize}"
        return dim

    def __repr__(self):
        """Return a string representation of the StocksSet. Mainly for Jupyter notebooks.

        Returns
        -------
        str
            String representation of the StocksSet.
        """
        header = self._get_sw_headers()
        dimensions = self._get_dxs_dimensions()

        with df_temporal_header(self._prices_df, header) as df:
            with pd.option_context("display.show_dimensions", False):
                original_string = repr(df)

        # add dimension
        string = f"{original_string}\nStocksSet [{dimensions}]"

        return string

    def _repr_html_(self):
        """Return an HTML representation of the StocksSet.

        Returns
        -------
        str
            HTML representation of the StocksSet.
        """
        header = self._get_sw_headers()
        dimensions = self._get_dxs_dimensions()

        # retrieve the original string
        with df_temporal_header(self._prices_df, header) as df:
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
