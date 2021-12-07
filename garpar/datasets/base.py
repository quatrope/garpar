import abc

import attr

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
class MarketMakerABC(metaclass=abc.ABCMeta):

    random_state = hparam(
        default=None, converter=np.random.default_rng, repr=False
    )

    __marker_maker_cls_config__ = {"repr": False, "frozen": True}

    # internal ================================================================

    def __init_subclass__(cls):
        """Initiate of subclasses.

        It ensures that every inherited class is decorated by ``attr.s()`` and
        assigns as class configuration the parameters defined in the class
        variable `__marker_maker_cls_config__`.

        In other words it is slightly equivalent to:

        .. code-block:: python

            @attr.s(**MarketMakerABC.__marker_maker_cls_config__)
            class Decomposer(MarketMakerABC):
                pass

        """
        model_config = getattr(cls, "__marker_maker_cls_config__")
        attr.s(maybe_cls=cls, **model_config)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        clsname = type(self).__name__

        selfd = attr.asdict(
            self,
            recurse=False,
            filter=lambda attr, _: attr.repr,
        )
        attrs_str = ", ".join([f"{k}={repr(v)}" for k, v in selfd.items()])
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
        timeserie = np.empty(days, dtype=float)
        for day, loss in enumerate(loss_sequence):
            current_price = self.make_stock_price(current_price, loss, random)
            timeserie[day] = current_price
        return pd.DataFrame({"price": timeserie})

    def make_market(
        self,
        *,
        window_size=5,
        days=365,
        entropy=0.5,
        stock_number=100,
        price=100,
    ):

        initial_prices = self._coerce_price(stock_number, price)

        window_loss_probability = self.get_window_loss_probability(
            window_size, entropy
        )

        stocks, stock_initial_prices = [], {}

        for stock_idx, stock_price in enumerate(initial_prices):
            stock_df = self.make_stock(
                days=days,
                window_size=window_size,
                window_loss_probability=window_loss_probability,
                initial_price=stock_price,
                random=self.random_state,
            )
            stock_df.rename(
                columns={"price": f"stock_{stock_idx}_price"}, inplace=True
            )
            stocks.append(stock_df)

            stock_initial_prices[f"stock_{stock_idx}"] = stock_price

        stock_df = pd.concat(stocks, axis=1)

        return pf.Portfolio.mkportfolio(
            stock_df,
            initial_prices=pd.Series(stock_initial_prices),
            entropy=entropy,
            window_size=window_size,
        )
