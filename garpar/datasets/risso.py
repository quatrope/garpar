# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

from dataclasses import dataclass, field

import numpy as np

import scipy.stats

from .base import RandomEntropyPortfolioMakerABC
from ..utils import mabc


# =============================================================================
# UTILS
# =============================================================================


def argnearest(arr, v):
    diff = np.abs(np.subtract(arr, v))
    idx = np.argmin(diff)
    return idx


# =============================================================================
# BASE
# =============================================================================


class RissoABC(RandomEntropyPortfolioMakerABC):
    def candidate_entropy(self, window_size):
        loss_probability = np.linspace(0.0, 1.0, num=window_size + 1)

        # Se corrigen probabilidades porque el cálculo de la entropía trabaja
        # con logaritmo y el logaritmo de cero no puede calcularse
        epsilon = np.finfo(loss_probability.dtype).eps
        loss_probability[0] = epsilon
        loss_probability[-1] = 1 - epsilon

        # Calcula entropy
        first_part = loss_probability * np.log2(loss_probability)
        second_part = (1 - loss_probability) * np.log2(1 - loss_probability)

        modificated_entropy = -1 * (first_part + second_part)
        return modificated_entropy, loss_probability

    def get_window_loss_probability(self, window_size, entropy):
        h_candidates, loss_probabilities = self.candidate_entropy(window_size)
        idx = argnearest(h_candidates, entropy)
        loss_probability = loss_probabilities[idx]

        return loss_probability


# =============================================================================
# NORMAL
# =============================================================================
class RissoUniform(RissoABC):
    low = mabc.hparam(default=1, converter=float)
    high = mabc.hparam(default=5, converter=float)

    def make_stock_price(self, price, loss, random):
        if price == 0.0:
            return 0.0
        sign = -1 if loss else 1
        day_return = sign * np.abs(random.uniform(self.low, self.high))
        new_price = price + day_return
        return 0.0 if new_price < 0 else new_price


def make_risso_uniform(
    low=1,
    high=5,
    *,
    entropy=0.5,
    random_state=None,
    n_jobs=None,
    verbose=0,
    **kwargs,
):
    maker = RissoUniform(
        low=low,
        high=high,
        entropy=entropy,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    port = maker.make_portfolio(**kwargs)
    return port


# =============================================================================
# NORMAL
# =============================================================================
class RissoNormal(RissoABC):
    mu = mabc.hparam(default=0, converter=float)
    sigma = mabc.hparam(default=0.2, converter=float)

    def make_stock_price(self, price, loss, random):
        if price == 0.0:
            return 0.0
        sign = -1 if loss else 1
        day_return = sign * np.abs(random.normal(self.mu, self.sigma))
        new_price = price + day_return
        return 0.0 if new_price < 0 else new_price


def make_risso_normal(
    mu=0,
    sigma=0.2,
    *,
    entropy=0.5,
    random_state=None,
    n_jobs=None,
    verbose=0,
    **kwargs,
):
    maker = RissoNormal(
        mu=mu,
        sigma=sigma,
        entropy=entropy,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    port = maker.make_portfolio(**kwargs)
    return port


# =============================================================================
# LEVY STABLE
# =============================================================================


@dataclass
class _LStableCache:
    negatives: list = field(default_factory=list, init=False, repr=False)
    positives: list = field(default_factory=list, init=False, repr=False)
    refresh_size: int = field(default=100, init=False)
    refresh: int = field(default=0, init=False)

    def __repr__(self):
        cname = type(self).__name__
        negs = len(self.negatives)
        pos = len(self.positives)
        total = negs + pos
        rsize = self.refresh_size
        refresh = self.refresh
        return (
            f"<{cname} (-, +, total)=({negs}, {pos}, {total}), "
            f"refresh_size={rsize}, refresh={refresh}>"
        )

    def get_value(self, sign, refresher, random):
        # we get the reference of the cache to use depending on the sign
        cache = self.negatives if sign < 0 else self.positives

        self.refresh = self.refresh + bool(cache)
        while not cache:
            values = refresher(size=self.refresh_size, random_state=random)

            # lo dividimos en positivos y negaticos
            self.negatives.extend(values[values < 0])
            self.positives.extend(values[values >= 0])

            # y la siguiente vez que se le pida valores, pedimos un poco maas
            self.refresh_size = self.refresh_size + int(
                np.log(10 + self.refresh_size)
            )

        return cache.pop(0)


class RissoLevyStable(RissoABC):
    alpha = mabc.hparam(default=1.6411, converter=float)
    beta = mabc.hparam(default=-0.0126, converter=float)
    mu = mabc.hparam(default=0.0005, converter=float)  # loc
    sigma = mabc.hparam(default=0.005, converter=float)  # scale

    levy_stable_ = mabc.mproperty(repr=False)

    _days_returns_cache = mabc.mproperty(repr=False, factory=_LStableCache)

    @levy_stable_.default
    def _levy_stable_default(self):
        return scipy.stats.levy_stable(
            alpha=self.alpha, beta=self.beta, loc=self.mu, scale=self.sigma
        )

    def make_stock_price(self, price, loss, random):
        if price == 0.0:
            return 0.0
        sign = -1 if loss else 1
        new_price = price + self._days_returns_cache.get_value(
            sign, self.levy_stable_.rvs, random
        )
        return 0.0 if new_price < 0 else new_price


def make_risso_levy_stable(
    alpha=1.6411,
    beta=-0.0126,
    mu=0.0005,
    sigma=0.005,
    *,
    entropy=0.5,
    random_state=None,
    n_jobs=None,
    verbose=0,
    **kwargs,
):
    maker = RissoLevyStable(
        alpha=alpha,
        beta=beta,
        mu=mu,
        sigma=sigma,
        entropy=entropy,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    port = maker.make_portfolio(**kwargs)
    return port
