import numpy as np
import pandas as pd


from .base import PortfolioMakerABC

from ..core.portfolio import Portfolio
from ..utils import mabc, Bunch


class MultiSector(PortfolioMakerABC):
    makers = mabc.hparam(converter=lambda v: tuple(dict(v).items()))

    @makers.validator
    def _makers_validator(self, attribute, value):
        if len(value) < 2:
            raise ValueError(f"You must provide at least 2 makers")
        for maker_name, maker in value:
            if not isinstance(maker, PortfolioMakerABC):
                cls_name = type(self).__name__
                msg = f"Maker '{maker_name}' is not instance of '{cls_name}'"
                raise TypeError(msg)

    def _coerce_price(self, stocks, prices, makers_len):
        if isinstance(prices, (int, float)):
            prices = np.full(stocks, prices, dtype=float)
        elif len(prices) != stocks:
            raise ValueError(f"The number of prices must be equal {stocks}")
        prices = np.asarray(prices, dtype=float)
        return np.array_split(prices, makers_len)

    def make_portfolio(
        self, *, window_size=5, days=365, stocks=10, price=100, weights=None
    ):
        makers_len = len(self.makers)
        if stocks < makers_len:
            raise ValueError(f"stocks must be >= {makers_len}")

        prices = self._coerce_price(stocks, price, makers_len)

        stocks_dfs = []
        entropy = []
        metadata = {}
        for (maker_name, maker), maker_prices in zip(self.makers, prices):
            stocks_number = len(maker_prices)
            pf = maker.make_portfolio(
                window_size=window_size,
                days=days,
                stocks=stocks_number,
                price=maker_prices,
                weights=None,
            )

            df = pf._df.add_prefix(f"{maker_name}_")
            stocks_dfs.append(df)

            entropy.extend(pf.entropy)

            metadata[maker_name] = Bunch(
                maker_name,
                {
                    "stocks": df.columns.to_numpy(),
                    "og_metadata": pf.metadata,
                    "stocks_number": stocks_number,
                },
            )

        # join all the dfs in one
        stock_df = pd.concat(stocks_dfs, axis="columns")

        # create the portfolio
        return Portfolio.from_dfkws(
            stock_df,
            weights=weights,
            window_size=window_size,
            entropy=entropy,
            **metadata,
        )
