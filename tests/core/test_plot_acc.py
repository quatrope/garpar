# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE


from garpar.core.plot_acc import StocksSetPlotterAccessor

from matplotlib.testing.decorators import check_figures_equal

import pytest

import seaborn as sns


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_line(
    risso_stocks_set, fig_test, fig_ref, returns, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.line(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    sns.lineplot(data=data, ax=ax_ref)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_heatmap(
    risso_stocks_set, fig_test, fig_ref, returns, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.heatmap(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.heatmap(data=data)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_wheatmap(
    risso_stocks_set, fig_test, fig_ref, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.wheatmap(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.heatmap(data=data.T)
    ax_ref.set_title(title)
    ax_ref.set_xlabel("Stocks")


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_hist(
    risso_stocks_set, fig_test, fig_ref, returns, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.hist(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.histplot(data=data)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_whist(
    risso_stocks_set, fig_test, fig_ref, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.whist(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.histplot(data=data.T)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_box(
    risso_stocks_set, fig_test, fig_ref, returns, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.box(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.boxplot(data=data)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_wbox(
    risso_stocks_set, fig_test, fig_ref, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.wbox(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.boxplot(data=data)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_kde(
    risso_stocks_set, fig_test, fig_ref, returns, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.kde(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.kdeplot(data=data)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_wkde(
    risso_stocks_set, fig_test, fig_ref, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.wkde(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.kdeplot(data=data)
    ax_ref.set_title(title)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("price_distribution", pytest.DISTRIBUTIONS)
def test_StocksSetPlotterAccessor_ogive(
    risso_stocks_set, fig_test, fig_ref, returns, price_distribution
):
    ss = risso_stocks_set(random_state=3, distribution=price_distribution)
    plotter = StocksSetPlotterAccessor(ss)

    ax_test = fig_test.subplots()
    plotter.ogive(returns=returns, ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=returns)
    ax_ref = sns.ecdfplot(data=data)
    ax_ref.set_title(title)
