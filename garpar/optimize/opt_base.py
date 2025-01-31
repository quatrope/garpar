# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021-2025 Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

from ..utils import mabc

# =============================================================================
# ABSTRACT OPTIMIZER
# =============================================================================

_Unknow = object()


class OptimizerABC(mabc.ModelABC):
    """
    Abstract base class for stocks set optimizers.

    Attributes
    ----------
    family : str
        The family of the optimizer.

    Methods
    -------
    optimize(ss)
        Optimize the given stocks set.
    get_optimizer_family()
        Get the family of the optimizer.
    """

    family = _Unknow

    def __init_subclass__(cls):
        """Check if the 'family' attribute is of type string."""
        if cls.family is _Unknow or not isinstance(cls.family, str):
            cls_name = cls.__name__
            raise TypeError(f"'{cls_name}.family' must be redefined as string")

    @mabc.abstractmethod
    def _calculate_weights(self, ss):
        """Boilerplate method to calculate stocks set weights."""
        raise NotImplementedError()

    def optimize(self, ss):
        """Optimize the given stocks set.

        Parameters
        ----------
        ss : StocksSet
            The stocks set to optimize.

        Returns
        -------
        StocksSet
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


class MeanVarianceFamilyMixin:
    """Mixin class for mean-variance family optimizers."""

    family = "mean-variance"
