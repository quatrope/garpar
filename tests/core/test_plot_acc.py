# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# TODO TESTEAR
import pytest
from garpar.datasets.risso import make_risso_uniform
from garpar.core.plot_acc import PortfolioPlotter

@pytest.fixture(scope="module")
def plot_fn():
    def _plot(type):
        pf = make_risso_uniform(random_state=42)
        plotter = PortfolioPlotter(pf)
        plotter.__call__(type)

    return _plot


def test_portfolio_plotter_line(plot_fn):
    plot_fn("line")

def test_portfolio_plotter_heatmap(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.heatmap()

def test_portfolio_plotter_wheatmap(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.wheatmap()

def test_portfolio_plotter_hist(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.hist()

def test_portfolio_whist(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.whist()

def test_portfolio_box(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.box()

def test_portfolio_wbox(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.wbox()

def test_portfolio_kde(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.kde()

def test_portfolio_wkde(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.wkde()

def test_portfolio_ogive(risso_portfolio):
    pf = risso_portfolio(random_state=42)

    plotter = PortfolioPlotter(pf)

    plotter.ogive()
