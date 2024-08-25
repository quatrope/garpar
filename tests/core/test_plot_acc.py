# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


from garpar.core.plot_acc import PortfolioPlotter

from matplotlib.testing.decorators import check_figures_equal

import pytest

import seaborn as sns

class TestPortfolioPlotter:
    @pytest.fixture(autouse=True)
    def setup_plotter(self, risso_portfolio, distribution):
        self.pf = risso_portfolio(random_state=3, distribution=distribution)
        self.plotter = PortfolioPlotter(self.pf)

    @check_figures_equal()
    @pytest.mark.parametrize("returns", [True, False])
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_line(self, fig_test, fig_ref, returns):
        ax_test = fig_test.subplots()
        self.plotter.line(returns=returns, ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._ddf(returns=returns)
        sns.lineplot(data=data, ax=ax_ref)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("returns", [True, False])
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_heatmap(self, fig_test, fig_ref, returns):
        ax_test = fig_test.subplots()
        self.plotter.heatmap(returns=returns, ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._ddf(returns=returns)
        ax_ref = sns.heatmap(data=data)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_wheatmap(self, fig_test, fig_ref):
        ax_test = fig_test.subplots()
        self.plotter.wheatmap(ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._wdf()
        ax_ref = sns.heatmap(data=data.T)
        ax_ref.set_title(title)
        ax_ref.set_xlabel("Stocks")


    @check_figures_equal()
    @pytest.mark.parametrize("returns", [True, False])
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_hist(self, fig_test, fig_ref, returns):
        ax_test = fig_test.subplots()
        self.plotter.hist(returns=returns, ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._ddf(returns=returns)
        ax_ref = sns.histplot(data=data)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_whist(self, fig_test, fig_ref):
        ax_test = fig_test.subplots()
        self.plotter.whist(ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._wdf()
        ax_ref = sns.histplot(data=data.T)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("returns", [True, False])
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_box(self, fig_test, fig_ref, returns):
        ax_test = fig_test.subplots()
        self.plotter.box(returns=returns, ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._ddf(returns=returns)
        ax_ref = sns.boxplot(data=data)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_wbox(self, fig_test, fig_ref):
        ax_test = fig_test.subplots()
        self.plotter.wbox(ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._wdf()
        ax_ref = sns.boxplot(data=data)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("returns", [True, False])
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_kde(self, fig_test, fig_ref, returns):
        ax_test = fig_test.subplots()
        self.plotter.kde(returns=returns, ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._ddf(returns=returns)
        ax_ref = sns.kdeplot(data=data)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_wkde(self, fig_test, fig_ref):
        ax_test = fig_test.subplots()
        self.plotter.wkde(ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._wdf()
        ax_ref = sns.kdeplot(data=data)
        ax_ref.set_title(title)


    @check_figures_equal()
    @pytest.mark.parametrize("returns", [True, False])
    @pytest.mark.parametrize("distribution", pytest.DISTRIBUTIONS)
    def test_ogive(self, fig_test, fig_ref, returns):
        ax_test = fig_test.subplots()
        self.plotter.ogive(returns=returns, ax=ax_test)

        ax_ref = fig_ref.subplots()
        data, title = self.plotter._ddf(returns=returns)
        ax_ref = sns.ecdfplot(data=data)
        ax_ref.set_title(title)
