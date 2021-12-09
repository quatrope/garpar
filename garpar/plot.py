# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import attr

# import matplotlib.pyplot as plt

import seaborn as sns


# =============================================================================
# PLOTTER OBJECT
# =============================================================================


@attr.s(frozen=True, cmp=False, slots=True)
class PortfolioPlotter:
    """Make plots of Portfolio."""

    _pf = attr.ib()

    # INTERNAL ================================================================

    def __call__(self, plot_kind="line", **kwargs):
        if plot_kind.startswith("_"):
            raise ValueError(f"invalid plot_kind name '{plot_kind}'")
        method = getattr(self, plot_kind, None)
        if not callable(method):
            raise ValueError(f"invalid plot_kind name '{plot_kind}'")
        return method(**kwargs)

    @property
    def _ddf(self):
        return self._pf._df

    @property
    def _stocks_labels(self):
        return self._pf._df.columns

    def line(self, **kwargs):
        ax = sns.lineplot(data=self._ddf, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._stocks_labels)
        return ax

    def heatmap(self, **kwargs):
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
        ax = sns.heatmap(self._ddf, **kwargs)
        ax.set_ylabel("Days")
        ax.set_xlabel("Stoks")
        return ax

    def hist(self, **kwargs):
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
        ax = sns.histplot(self._ddf, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._stocks_labels)
        return ax

    def box(self, **kwargs):
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
        ax = sns.boxplot(data=self._ddf, **kwargs)
        return ax

    def violin(self, **kwargs):
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
        ax = sns.violinplot(data=self._ddf, **kwargs)
        return ax

    def boxen(self, **kwargs):
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
        ax = sns.boxenplot(data=self._ddf, **kwargs)
        return ax

    def kde(self, **kwargs):
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
        ax = sns.kdeplot(data=self._ddf, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._stocks_labels)
        return ax