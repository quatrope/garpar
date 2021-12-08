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

# import seaborn as sns


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

    def __getattr__(self, a):
        plot_df = self._pf._df.plot
        return getattr(plot_df, a)
