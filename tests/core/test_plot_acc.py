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

@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_line(risso_portfolio, fig_test, fig_ref, returns, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.line(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    sns.lineplot(data=data, ax=ax_ref)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_heatmap(risso_portfolio, fig_test, fig_ref, returns, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.heatmap(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.heatmap(data=data)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_wheatmap(risso_portfolio, fig_test, fig_ref, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.wheatmap(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.heatmap(data=data.T)
    ax_ref.set_title(title)
    ax_ref.set_xlabel("Stocks")


@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_hist(risso_portfolio, fig_test, fig_ref, returns, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.hist(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.histplot(data=data)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_whist(risso_portfolio, fig_test, fig_ref, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.whist(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.histplot(data=data.T)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_box(risso_portfolio, fig_test, fig_ref, returns, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.box(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.boxplot(data=data)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_wbox(risso_portfolio, fig_test, fig_ref, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.wbox(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.boxplot(data=data)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_kde(risso_portfolio, fig_test, fig_ref, returns, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.kde(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.kdeplot(data=data)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_wkde(risso_portfolio, fig_test, fig_ref, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.wkde(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.kdeplot(data=data)
    ax_ref.set_title(title)


@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_PortFolioPlotter_ogive(risso_portfolio, fig_test, fig_ref, returns, price_distribution):
    pf = risso_portfolio(random_state=3, distribution=price_distribution)
    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.ogive(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.ecdfplot(data=data)
    ax_ref.set_title(title)
