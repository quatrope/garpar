# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import attr

import seaborn as sns

from ..utils import accabc


# =============================================================================
# PLOTTER OBJECT
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True, repr=False)
class PortfolioPlotter(accabc.AccessorABC):
    """Make plots of Portfolio."""

    _default_kind = "line"

    _pf = attr.ib()

    # INTERNAL ================================================================

    def _ddf(self, returns):
        if returns:
            return self._pf.as_returns(), "Returns"
        return self._pf._prices_df, "Price"

    def _wdf(self):
        # proxy to access the dataframe with the weights
        return self._pf.weights.to_frame(), "Weights"
    
    def _edf(self):
        return self._pf.entropy.to_frame(), "Entropy"

    # PLOTS ===================================================================

    def line(self, returns=False, **kwargs):
        data, title = self._ddf(returns=returns)
        ax = sns.lineplot(data=data, **kwargs)
        ax.set_title(title)
        return ax

    def heatmap(self, returns=False, **kwargs):
        """Plot the portfolio matrix as a color-encoded matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.heatmap``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._ddf(returns=returns)

        ax = sns.heatmap(data=data, **kwargs)
        ax.set_title(title)
        ax.set_ylabel("Days")
        ax.set_xlabel("Stocks")

        return ax

    def wheatmap(self, **kwargs):
        """Plot weights as a color-encoded matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.heatmap``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._wdf()
        ax = sns.heatmap(data=data.T, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("Stocks")

        if "ax" not in kwargs:
            # if the ax is provided by the user we assume that the figure
            # is already setted to the expected size. If it's not we resize the
            # height to 1/5 of the original size.
            fig = ax.get_figure()
            size = fig.get_size_inches() / [1, 5]
            fig.set_size_inches(size)

        return ax

    def hist(self, returns=False, **kwargs):
        """Draw one histogram of the portfolio.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._ddf(returns=returns)
        ax = sns.histplot(data=data, **kwargs)
        ax.set_title(title)
        return ax

    def whist(self, **kwargs):
        """Draw one histogram of the weights.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._wdf()
        ax = sns.histplot(data=data.T, **kwargs)
        ax.set_title(title)
        return ax

    def box(self, returns=False, **kwargs):
        """Make a box plot of the portfolio.

        A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.

        For further details see Wikipedia's
        entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.boxplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._ddf(returns=returns)
        ax = sns.boxplot(data=data, **kwargs)
        ax.set_title(title)
        return ax

    def wbox(self, **kwargs):
        """Make a box plot of the weights.

        A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.

        For further details see Wikipedia's
        entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.boxplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._wdf()
        ax = sns.boxplot(data=data**kwargs)
        ax.set_title(title)
        return ax

    def kde(self, returns=False, **kwargs):
        """Stock kernel density plot using Gaussian kernels.

        In statistics, `kernel density estimation`_ (KDE) is a non-parametric
        way to estimate the probability density function (PDF) of a random
        variable. This function uses Gaussian kernels and includes automatic
        bandwidth determination.

        .. _kernel density estimation:
            https://en.wikipedia.org/wiki/Kernel_density_estimation

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._ddf(returns=returns)
        ax = sns.kdeplot(data=data, **kwargs)
        ax.set_title(title)
        return ax

    def wkde(self, **kwargs):
        """Weights kernel density plot using Gaussian kernels.

        In statistics, `kernel density estimation`_ (KDE) is a non-parametric
        way to estimate the probability density function (PDF) of a random
        variable. This function uses Gaussian kernels and includes automatic
        bandwidth determination.

        .. _kernel density estimation:
            https://en.wikipedia.org/wiki/Kernel_density_estimation

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        data, title = self._wdf()
        ax = sns.kdeplot(data=data**kwargs)
        ax.set_title(title)
        return ax

    def ogive(self, returns=False, **kwargs):
        data, title = self._ddf(returns=returns)
        ax = sns.ecdfplot(data=data, **kwargs)
        ax.set_title(title)
        return ax
