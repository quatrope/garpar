# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

import pytest
from garpar.datasets.risso import make_risso_uniform
from garpar.core.portfolio import Portfolio
from garpar.core.plot_acc import PortfolioPlotter
from matplotlib.testing.decorators import check_figures_equal
import seaborn as sns
import numpy as np

@check_figures_equal()
def test_portfolio_plotter_line(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.line(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=False)
    ax_ref = sns.lineplot(data=data)
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_plotter_heatmap(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.heatmap(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=False)
    ax_ref = sns.heatmap(data=data)
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_plotter_wheatmap(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.wheatmap(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.heatmap(data=data.T)
    ax_ref.set_title(title)
    ax_ref.set_xlabel("Stocks")

@check_figures_equal()
def test_portfolio_plotter_hist(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.hist(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=False)
    ax_ref = sns.histplot(data=data)
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_whist(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.whist(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.histplot(data=data.T)
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_box(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.box(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=False)
    ax_ref = sns.boxplot(data=data)
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_wbox(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.wbox(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.boxplot(data=data)
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_kde(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.kde(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=False)
    ax_ref = sns.kdeplot(data=data)
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_wkde(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.wkde(ax=ax_test, warn_singular=False) # FIXME preguntar si es correcto

    ax_ref = fig_ref.subplots()
    data, title = plotter._wdf()
    ax_ref = sns.kdeplot(data=data, warn_singular=False) # FIXME preguntar si es correcto
    ax_ref.set_title(title)

@check_figures_equal()
def test_portfolio_ogive(fig_test, fig_ref):
    pf = make_risso_uniform(random_state=3)

    plotter = PortfolioPlotter(pf)

    ax_test = fig_test.subplots()
    plotter.ogive(ax=ax_test)

    ax_ref = fig_ref.subplots()
    data, title = plotter._ddf(returns=False)
    ax_ref = sns.ecdfplot(data=data)
    ax_ref.set_title(title)
