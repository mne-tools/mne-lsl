"""
Module for signal acquisition.

Acquire signals from LSL streams. Used by other modules such as StreamViewer.
"""

from ._stream import StreamEEG, StreamMarker
from .stream_receiver import StreamReceiver

__all__ = ["StreamReceiver", "StreamEEG", "StreamMarker"]
