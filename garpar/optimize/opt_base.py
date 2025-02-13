# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Optimizer base classes module.

This module provides an abstract base class for optimization models. It
also provides a mixin class for mean-variance family optimizers.

Example:
    >>> from garpar.optimize.opt_base import OptimizerABC
    >>> class MyFamilyMixin:
            family = "AnOptimizerFamily"
    >>> class MyOptimizer(MyFamilyMixin, OptimizerABC):
    ...     family = "AnOptimizerFamily"
    ...     def _calculate_weights(self, ss):
    ...         return ss.weights/len(ss.weights), {}
    >>> my_optimizer = MyOptimizer()
    >>> ss = garpar.mkss(prices=[...])
    >>> my_optimizer.optimize(ss)

"""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import mabc

# =============================================================================
# CONSTANTS
# =============================================================================

_Unknow = object()

# =============================================================================
# ABSTRACT OPTIMIZER
# =============================================================================


class OptimizerABC(mabc.ModelABC):
    """Abstract base class for stocks set optimizers.

    This abstract class defines which methods should an optimizer have.
    """

    family = _Unknow

    def __init_subclass__(cls):
        """Check if the 'family' attribute is of type string."""
        if cls.family is _Unknow or not isinstance(cls.family, str):
            cls_name = cls.__name__
            raise TypeError(f"'{cls_name}.family' must be redefined as string")

    @mabc.abstractmethod
    def _calculate_weights(self, ss):
        """Abstract method to calculate stocks set weights."""
        raise NotImplementedError()

    def optimize(self, ss):
        """Optimize the given stocks set.

        Parameters
        ----------
        ss: garpar.core.stocks_set.StocksSet
            The stocks set to optimize.

        Returns
        -------
        garpar.core.stocks_set.StocksSet
            A new stocks set with optimized weights.
        """
        weights, metadata = self._calculate_weights(ss)
        return ss.copy(weights=weights, optimizer=metadata)

    @classmethod
    def get_optimizer_family(cls):
        """Get the family of the optimizer.

        Returns
        -------
        str
            The family of the optimizer.
        """
        return cls.family


# =============================================================================
# MEAN VARIANCE FAMILY MIXIN
# =============================================================================


class MeanVarianceFamilyMixin:
    """Mixin class for mean-variance like optimizers."""

    family = "mean-variance"
