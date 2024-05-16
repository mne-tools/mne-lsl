from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..utils._docs import fill_doc

if TYPE_CHECKING:
    from typing import Any


@fill_doc
class BaseEpochsStream(ABC):
    """Stream object representing a single real-time stream of epochs.

    Note that a stream of epochs is necessarily connected to a regularly sampled stream
    of continuous data, from which epochs are extracted depending on an internal event
    channel or to an external event stream.

    Parameters
    ----------
    bufsize : int
        Number of epochs to keep in the buffer.
    event_source : Any
    %(epochs_tmin_tmax)s
    %(baseline_epochs)s
    %(reject_epochs)s
    %(flat)s
    %(epochs_reject_tmin_tmax)s
    detrend : int | str | None
        The type of detrending to use. Can be ``'constant'`` or ``0`` for constant (DC)
        detrend, ``'linear'`` or ``1`` for linear detrend, or ``None`` for no
        detrending. Note that detrending is performed before baseline correction.
    """

    @abstractmethod
    def __init__(
        self,
        bufsize: float,
        event_source: Any,
        tmin: float = -0.2,
        tmax: float = 0.5,
        baseline: tuple[float | None, float | None] | None = (None, 0),
        reject: dict[str, float] | None = None,
        flat: dict[str, float] | None = None,
        reject_tmin: float | None = None,
        reject_tmax: float | None = None,
        detrend: int | str | None = None,
    ) -> None:
        pass
