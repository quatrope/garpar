# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
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
    Abstract base class for portfolio optimizers.

    Attributes
    ----------
    family : str
        The family of the optimizer.

    Methods
    -------
    optimize(pf)
        Optimize the given portfolio.
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
    def _calculate_weights(self, pf):
        """Boilerplate method to calculate portfolio weights."""
        raise NotImplementedError()

    def optimize(self, pf):
        """Optimize the given portfolio.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        Portfolio
            A new portfolio with optimized weights.
        """
        weights, metadata = self._calculate_weights(pf)
        return pf.copy(weights=weights, optimizer=metadata)

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
