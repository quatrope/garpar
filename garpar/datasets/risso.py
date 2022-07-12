# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np

import scipy.stats

from .base import PortfolioMakerABC
from ..utils.mabc import hparam, mproperty


# =============================================================================
# UTILS
# =============================================================================


def argnearest(arr, v):
    diff = np.abs(np.subtract(arr, v))
    idx = np.argmin(diff)
    return idx


# =============================================================================
# NORMAL
# =============================================================================


class RissoABC(PortfolioMakerABC):
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


class RissoNormal(RissoABC):

    mu = hparam(default=0, converter=float)
    sigma = hparam(default=0.2, converter=float)

    def make_stock_price(self, price, loss, random):
        if price == 0.0:
            return 0.0
        sign = -1 if loss else 1
        day_return = sign * np.abs(random.normal(self.mu, self.sigma))
        new_price = price + day_return
        return 0.0 if new_price < 0 else new_price


class RissoLevyStable(RissoABC):

    alpha = hparam(default=1.6411, converter=float)
    beta = hparam(default=-0.0126, converter=float)
    mu = hparam(default=0.0005, converter=float)  # loc
    sigma = hparam(default=0.005, converter=float)  # scale

    levy_stable_ = mproperty(repr=False)

    @levy_stable_.default
    def _levy_stable_default(self):
        return scipy.stats.levy_stable(
            alpha=self.alpha, beta=self.beta, loc=self.mu, scale=self.sigma
        )

    def make_stock_price(self, price, loss, random):
        if price == 0.0:
            return 0.0
        sign = -1 if loss else 1
        day_return = sign * np.abs(self.levy_stable_.rvs(random_state=random))
        new_price = price + day_return
        return 0.0 if new_price < 0 else new_price
