"""Events page: stim channels of the stream and attached LSL event streams.

Structured placeholder. The stim-channel section reads the stream itself and detects
edges, while the 'LSL Event' section is an accepted protocol-specific seam: a string or
irregularly sampled marker stream cannot be consumed through
:class:`~mne_lsl.stream.BaseStream` and needs the low-level
:class:`~mne_lsl.lsl.StreamInlet`, which is future work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    from ...stream import BaseStream


class EventsPage(QWidget):
    """Events page of the controller.

    Two labelled sections:

    - **Stim channels**, listing the stim channels of this stream, with an enable toggle
      and the edge/value semantics used to turn samples into events;
    - **LSL Event**, listing the attached irregular or string LSL streams, with their
      identity, connection state, interpretation mode and a remove action.

    Overlays are attached per document, because the timing alignment depends on the
    display processing of that document.

    Parameters
    ----------
    stream : BaseStream
        The connected stream the overlays are aligned to.
    parent : QWidget | None
        Parent widget.
    """

    def __init__(self, stream: BaseStream, parent: QWidget | None = None) -> None:
        """Initialize the page."""
