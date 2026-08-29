"""Processing page: reference, detrending and filtering.

Structured placeholder. The functionality depends on stream-subsystem extensions which
do not exist yet and which are in scope of a later phase: an online phase-aligned FIR
filter exposing its group delay, an in-place re-referencing which does not require a
reconnection, and a whole-buffer detrending operation. Until then, this page defines
where those controls live and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    from ...stream import BaseStream


class ProcessingPage(QWidget):
    """Processing page of the controller.

    Planned sections, in the order the pipeline applies them: reference, i.e. none,
    common average or selected channels; detrending; and filtering, offering both a
    phase-aligned FIR with a displayed latency and the current causal IIR. Applying a
    change is transactional: it rebuilds the pipeline, clears the visible history,
    enters a warm-up state and then reports the effective display latency.

    Parameters
    ----------
    stream : BaseStream
        The connected stream the processing is applied to.
    parent : QWidget | None
        Parent widget.
    """

    def __init__(self, stream: BaseStream, parent: QWidget | None = None) -> None:
        """Initialize the page."""

    def apply(self) -> None:
        """Apply the requested processing to the stream, transactionally."""
