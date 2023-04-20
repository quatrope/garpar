import attr

from abc import ABCMeta, abstractmethod  # noqa

HPARAM_METADATA_FLAG = "__hparam__"

MPROPERTY_METADATA_FLAG = "__mproperty__"

MODEL_CONFIG = "__model_cls_config__"


def hparam(**kwargs):
    """Create a hyper parameter for market maker.

    By design decision, hyper-parameter is required to have a sensitive default
    value.

    Parameters
    ----------
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
    metadata[HPARAM_METADATA_FLAG] = True
    return attr.ib(
        metadata=metadata,
        kw_only="default" in kwargs,
        **kwargs,
    )


def mproperty(**kwargs):
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
    metadata[MPROPERTY_METADATA_FLAG] = True
    return attr.ib(init=False, metadata=metadata, **kwargs)


@attr.s(repr=False)
class ModelABC(metaclass=ABCMeta):
    __model_cls_config__ = {"repr": False, "frozen": True}

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
        model_config = getattr(cls, MODEL_CONFIG)
        acls = attr.s(maybe_cls=cls, **model_config)

        return acls

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        clsname = type(self).__name__

        selfd = attr.asdict(
            self,
            recurse=False,
            filter=lambda attr, _: (
                attr.metadata.get(HPARAM_METADATA_FLAG) and attr.repr
            ),
        )
        hparams = sorted(selfd.items())
        attrs_str = ", ".join([f"{k}={v!r}" for k, v in hparams])
        return f"{clsname}({attrs_str})"
