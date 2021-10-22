"""
Module for signal acquisition.

Acquire signals from LSL streams. Used by other modules such as StreamViewer
and StreamRecorder.
"""

from ._stream import StreamMarker, StreamEEG  # noqa: F401
from .stream_receiver import StreamReceiver  # noqa: F401
