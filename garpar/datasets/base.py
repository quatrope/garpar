# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================
import abc

import attr

import joblib

import numpy as np

import pandas as pd

from .. import portfolio as pf

# =============================================================================
# BASE
# =============================================================================


def hparam(default, **kwargs):
    """Create a hyper parameter for market maker.

    By design decision, hyper-parameter is required to have a sensitive default
    value.

    Parameters
    ----------
    default :
        Sensitive default value of the hyper-parameter.
    **kwargs :
        Additional keyword arguments are passed and are documented in
        ``attr.ib()``.

    Return
    ------
    Hyper parameter with a default value.

    Notes
    -----
    This function is a thin-wrapper over the attrs function ``attr.ib()``.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["__garpar_model_hparam__"] = True
    return attr.ib(default=default, metadata=metadata, kw_only=True, **kwargs)


@attr.s(frozen=True, repr=False)
class PortfolioMakerABC(metaclass=abc.ABCMeta):

    random_state = hparam(
        default=None, converter=np.random.default_rng, repr=False
    )
    n_jobs = hparam(default=None)
    verbose = hparam(default=0)

    __portfolio_maker_cls_config__ = {"repr": False, "frozen": True}

    # internal ================================================================

    def __init_subclass__(cls):
        """Initiate of subclasses.

        It ensures that every inherited class is decorated by ``attr.s()`` and
        assigns as class configuration the parameters defined in the class
        variable `__portfolio_maker_cls_config__`.

        In other words it is slightly equivalent to:

        .. code-block:: python

            @attr.s(**PortfolioMakerABC.__portfolio_maker_cls_config__)
            class Decomposer(PortfolioMakerABC):
                pass

        """
        model_config = getattr(cls, "__portfolio_maker_cls_config__")
        attr.s(maybe_cls=cls, **model_config)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        clsname = type(self).__name__

        selfd = attr.asdict(
            self,
            recurse=False,
            filter=lambda attr, _: attr.repr,
        )
        hparams = sorted(selfd.items())
        attrs_str = ", ".join([f"{k}={repr(v)}" for k, v in hparams])
        return f"{clsname}({attrs_str})"

    # Internal ================================================================

    def _coerce_price(self, stock_number, prices):
        if isinstance(prices, (int, float)):
            prices = np.full(stock_number, prices, dtype=float)
        elif len(prices) != stock_number:
            raise ValueError(f"The q of prices must be equal {stock_number}")
        return np.asarray(prices, dtype=float)

    # Abstract=================================================================

    @abc.abstractmethod
    def get_window_loss_probability(self, windows_size, entropy):
        raise NotImplementedError()

    @abc.abstractmethod
    def make_stock_price(self, price, loss, random):
        raise NotImplementedError()

    # API =====================================================================

    def get_loss_sequence(self, days, loss_probability, random):
        win_probability = 1 - loss_probability

        # primero seleccionamos con las probabilidades adecuadas si en cada
        # dia se pierde o se gana
        sequence = random.choice(
            [True, False],
            size=days,
            p=[loss_probability, win_probability],
        )

        # con esto generamos lugares al azar donde vamos a invertir la
        # secuencia anterior dado que las probabilidades representan
        # una distribucion simétrica
        reverse_mask = random.choice([True, False], size=days)

        # finalmente invertimos esos lugares
        final_sequence = np.where(reverse_mask, ~sequence, sequence)

        return final_sequence

    def make_stock(
        self,
        *,
        window_size,
        days,
        window_loss_probability,
        initial_price,
        random,
    ):

        loss_sequence = self.get_loss_sequence(
            days, window_loss_probability, random
        )
        current_price = initial_price
        timeserie = np.empty(days + 1, dtype=float)

        # first price is the initial one
        timeserie[0] = current_price

        for day, loss in enumerate(loss_sequence, start=1):
            current_price = self.make_stock_price(current_price, loss, random)
            timeserie[day] = current_price

        return pd.DataFrame({"price": timeserie})

    def _make_portfolio_stock(
        self,
        days,
        window_size,
        window_loss_probability,
        stock_idx,
        initial_price,
        seed,
    ):
        stock_df = self.make_stock(
            days=days,
            window_size=window_size,
            window_loss_probability=window_loss_probability,
            initial_price=initial_price,
            random=np.random.default_rng(seed),
        )
        stock_df.rename(columns={"price": f"stock_{stock_idx}"}, inplace=True)
        return stock_df

    def make_portfolio(
        self,
        *,
        window_size=5,
        days=365,
        entropy=0.5,
        stock_number=10,
        price=100,
    ):

        initial_prices = self._coerce_price(stock_number, price)

        window_loss_probability = self.get_window_loss_probability(
            window_size, entropy
        )

        iinfo = np.iinfo(int)
        seeds = self.random_state.integers(
            low=0,
            high=iinfo.max,
            size=stock_number,
            dtype=iinfo.dtype,
            endpoint=True,
        )
        idx_prices_seed = zip(range(stock_number), initial_prices, seeds)

        with joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="processes",
        ) as P:
            dmaker = joblib.delayed(self._make_portfolio_stock)
            stocks = P(
                dmaker(
                    days=days,
                    window_size=window_size,
                    window_loss_probability=window_loss_probability,
                    stock_idx=stock_idx,
                    initial_price=stock_price,
                    seed=seed,
                )
                for stock_idx, stock_price, seed in idx_prices_seed
            )

        stock_df = pd.concat(stocks, axis=1)

        return pf.Portfolio.from_dfkws(
            stock_df,
            entropy=entropy,
            window_size=window_size,
        )
