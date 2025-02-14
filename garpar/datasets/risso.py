# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Risso StocksSet Makers.

This module provides generators that use the Risso informational entropy
calculation to determine loss probabilities. Each generator also adheres to a
specific distribution for generating stock prices.

Key Features:
    - Entropy-based portfolio simulation

Examples
--------
    >>> import garpar
    >>> ss = garpar.datasets.make_risso_normal(stocks=2, days=20)
    >>> ss.as_prices()
    >>> ss.as_returns()

    or

    >>> from garpar.datasets import RissoNormal
    >>> maker = RissoNormal(
    ...     mu=10,
    ...     sigma=0.2,
    ...     entropy=0.5,
    ...     random_state=10,
    ...     n_jobs=None,
    ...     verbose=0
    ... )
    >>> maker.make_stocks_set()

References
----------
    Risso, W. A. (2008). The informational efficiency and the
    financial crashes.
    https://doi.org/10.1016/j.ribaf.2008.02.005.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from dataclasses import dataclass, field

import numpy as np

import scipy.stats

from .ds_base import RandomEntropyStocksSetMakerABC
from ..constants import EPSILON
from ..utils import mabc


# =============================================================================
# UTILS
# =============================================================================


def argnearest(arr, v):
    """Find the index of the element in the array `arr` that is closest to `v`.

    Parameters
    ----------
    arr : array_like
        Input array.
    v : scalar
        Value to which the elements of `arr` will be compared.

    Returns
    -------
    idx : int
        Index of the element in `arr` that is closest to `v`.

    Notes
    -----
    If there are multiple elements at the same distance from `v`, the index of
    the first occurrence is returned.

    Examples
    --------
    >>> arr = np.array([1, 3, 5, 7, 9])
    >>> argnearest(arr, 4)
    1
    >>> argnearest(arr, 6)
    2
    >>> argnearest(arr, 8)
    4
    """
    diff = np.abs(np.subtract(arr, v))
    idx = np.argmin(diff)
    return idx


# =============================================================================
# BASE
# =============================================================================


class RissoMixin:
    """Implementation of a stock set maker using Risso's entropy calculation.

    This class extends RandomEntropyStocksSetMakerABC and implements methods
    for calculating candidate entropies and selecting loss probabilities based
    on a given window size and target entropy.
    """

    def _generate_loss_probabilities(self, window_size, eps=None):
        """Calculate candidate entropies and corresponding loss probabilities.

        Note that for a greater window size, the chances of losing or winning
        are more transparent.

        Parameters
        ----------
        window_size : int
            Size of the sliding window for entropy calculation.

        Returns
        -------
        tuple
            Tuple containing the calculated modified entropy values and
            corresponding loss probabilities.
        """
        # log(0) is undefined
        epsilon = EPSILON if eps is None else eps

        loss_probability = np.linspace(
            epsilon, 1.0 - epsilon, num=window_size + 1
        )

        # calculate entropy with log2 as Risso with returns
        first_part = loss_probability * np.log2(loss_probability)
        second_part = (1.0 - loss_probability) * np.log2(
            1.0 - loss_probability
        )

        modificated_entropy = -1.0 * (first_part + second_part)
        return modificated_entropy, loss_probability

    def get_window_loss_probability(self, window_size, entropy, eps=None):
        """Get the loss probability that corresponds to the nearest entropy.

        Get the loss probability that corresponds to the nearest candidate
        entropy value to the target entropy.

        Parameters
        ----------
        window_size : int
            Size of the sliding window for entropy calculation.
        entropy : float
            Target entropy value for portfolio optimization.

        Returns
        -------
        float
            Loss probability that corresponds to the nearest candidate entropy
            value to the target entropy.

        Example
        -------

        >>> get_window_loss_probability(window_size=3, entropy=0.99)
        ... 0.33333333366666673

        """
        h_candidates, loss_probabilities = self._generate_loss_probabilities(
            window_size, eps
        )
        idx = argnearest(h_candidates, entropy)
        loss_probability = loss_probabilities[idx]

        return loss_probability


# =============================================================================
# UNIFORM
# =============================================================================


class RissoUniform(RissoMixin, RandomEntropyStocksSetMakerABC):
    """Implementation of a StocksSets maker using a uniform distribution.

    This class extends RissoABC and overrides the method make_stock_price to
    simulate stock price changes based on a uniform distribution within
    specified bounds.

    Notes
    -----
    Inherits from RissoABC and utilizes its methods for entropy-based portfolio
    optimization.
    """

    low = mabc.hparam(default=1, converter=float)
    high = mabc.hparam(default=5, converter=float)

    def make_stock_price(self, price, loss, random):
        """Calculate the new stock price.

        Calculate the new stock price based on the current price, loss flag,
        and a random number generator following a uniform distribution.

        Parameters
        ----------
        price : float
            Current price of the stock.
        loss : bool
            Flag indicating if it's a loss day (True) or a gain day (False).
        random : numpy.random.Generator
            Random number generator instance.

        Returns
        -------
        float
            New price of the stock after simulating the daily return.
        """
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
    """Create a StocksSet instance using RissoUniform maker.

    This function is an implementation of the factory method that creates a
    StocksSet using the RissoUniform maker.

    Parameters
    ----------
    low : float, optional
        Lower bound of the uniform distribution for daily returns.
        Default is 1.0.
    high : float, optional
        Upper bound of the uniform distribution for daily returns.
        Default is 5.0.
    entropy : float, optional
        Entropy parameter controlling the randomness in portfolio creation.
        Default is 0.5.
    random_state : {None, int, numpy.random.Generator}, optional
        Seed or Generator for the random number generator. Default is None.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is None.
    verbose : int, optional
        Verbosity level. Default is 0.
    **kwargs
        Additional keyword arguments passed to the make_stocks_set method
        of RissoUniform.

    Returns
    -------
    garpar.core.stocks_set.StocksSet
        Generated portfolio instance.

    Notes
    -----
    This function initializes a RissoUniform portfolio maker with specified
    parameters, generates a portfolio using those parameters, and returns the
    resulting portfolio object.
    """
    maker = RissoUniform(
        low=low,
        high=high,
        entropy=entropy,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return maker.make_stocks_set(**kwargs)


# =============================================================================
# NORMAL
# =============================================================================


class RissoNormal(RissoMixin, RandomEntropyStocksSetMakerABC):
    """Implementation of a stocks set maker using a normal distribution.

    Notes
    -----
    This class extends RissoABC and implements a portfolio maker using a
    normal distribution model for daily returns. The make_stock_price method
    generates stock prices based on the normal distribution parameters
    (mu, sigma).
    """

    mu = mabc.hparam(default=0, converter=float)
    sigma = mabc.hparam(default=0.2, converter=float)

    def make_stock_price(self, price, loss, random):
        """Generate a new stock price.

        Generate a new stock price based on current price, daily return
        direction, and normal distribution parameters.

        Parameters
        ----------
        price : float
            Current price of the stock.
        loss : bool
            Flag indicating if it's a loss day (True) or gain day (False).
        random : numpy.random.Generator
            Random number generator instance.

        Returns
        -------
        float
            New price of the stock after daily price change.

        Notes
        -----
        This method calculates a new stock price based on the current price,
        the direction of daily return (loss or gain), and the parameters of
        the normal distribution (mu, sigma).
        """
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
    """Create a stocks set using RissoNormal stocks set maker.

    Parameters
    ----------
    mu : float, optional
        Mean of the normal distribution for daily returns. Default is 0.0.
    sigma : float, optional
        Standard deviation of the normal distribution for daily returns.
        Default is 0.2.
    entropy : float, optional
        Entropy parameter controlling the randomness in stocks set creation.
        Default is 0.5.
    random_state : {None, int, numpy.random.Generator}, optional
        Seed or Generator for the random number generator. Default is None.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is None.
    verbose : int, optional
        Verbosity level. Default is 0.
    **kwargs
        Additional keyword arguments passed to the make_stocks_set method
        of RissoNormal.

    Returns
    -------
    garpar.core.stocks_set.StocksSet
        Generated stocks set instance.

    Notes
    -----
    This function initializes a RissoNormal stocks set maker with specified
    parameters, generates a stocks set using those parameters, and returns
    the resulting stocks set object.
    """
    maker = RissoNormal(
        mu=mu,
        sigma=sigma,
        entropy=entropy,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return maker.make_stocks_set(**kwargs)


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

            # split it into possitive and negatives
            self.negatives.extend(values[values < 0])
            self.positives.extend(values[values >= 0])

            # next time ask for more samples
            self.refresh_size = self.refresh_size + int(
                np.log(10 + self.refresh_size)
            )

        return cache.pop(0)


class RissoLevyStable(RissoMixin, RandomEntropyStocksSetMakerABC):
    """Implementation of a stocks set maker using Levy stable distribution.

    StocksSet maker implementing a stochastic model with Levy stable
    distribution for daily returns.

    Notes
    -----
    This class extends RissoABC and implements a stocks set maker using Levy
    stable distribution model for daily returns. The make_stock_price
    method generates stock prices based on the Levy stable distribution
    parameters (alpha, beta, mu, sigma).
    """

    alpha = mabc.hparam(default=1.6411, converter=float)  # shape
    beta = mabc.hparam(default=-0.0126, converter=float)  # scale
    mu = mabc.hparam(default=0.0005, converter=float)  # loc
    sigma = mabc.hparam(default=0.005, converter=float)  # scale

    levy_stable_ = mabc.mproperty(repr=False)

    _days_returns_cache = mabc.mproperty(repr=False, factory=_LStableCache)

    @levy_stable_.default
    def _levy_stable_default(self):
        """Initialize the Levy stable distribution object.

        Returns
        -------
        scipy.stats._continuous_distns.levy_stable_gen
            Levy stable distribution object initialized with the specified
            parameters.
        """
        return scipy.stats.levy_stable(
            alpha=self.alpha, beta=self.beta, loc=self.mu, scale=self.sigma
        )

    def make_stock_price(self, price, loss, random):
        """Generate a new stock price.

        Generate a new stock price based on current price, daily return
        direction, and Levy stable distribution parameters.

        Parameters
        ----------
        price : float
            Current price of the stock.
        loss : bool
            Flag indicating if it's a loss day (True) or gain day (False).
        random : numpy.random.Generator
            Random number generator instance.

        Returns
        -------
        float
            New price of the stock after daily price change.

        Notes
        -----
        This method calculates a new stock price based on the current price,
        the direction of daily return (loss or gain), and the parameters of
        the Levy stable distribution (alpha, beta, mu, sigma).
        """
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
    """Create a stocks set using the RissoLevyStable stocks set maker.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter of the Levy stable distribution. Default is 1.6411.
    beta : float, optional
        Scale parameter of the Levy stable distribution. Default is -0.0126.
    mu : float, optional
        Location parameter (mean) of the Levy stable distribution.
        Default is 0.0005.
    sigma : float, optional
        Scale parameter (spread) of the Levy stable distribution.
        Default is 0.005.
    entropy : float, optional
        Entropy parameter controlling the randomness in stocks set creation.
        Default is 0.5.
    random_state : {None, int, numpy.random.Generator}, optional
        Seed or Generator for the random number generator. Default is None.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is None.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    garpar.core.stocks_set.StocksSet
        StocksSet object representing the created stocks set.

    Notes
    -----
    This function initializes a RissoLevyStable stocks set maker with the
    provided parameters and creates a stocks set using the
    make_stocks_set method.
    """
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

    return maker.make_stocks_set(**kwargs)
