"""
Module for signal acquisition.

Acquire signals from LSL streams, used by other modules such as StreamViewer
and StreamRecorder.
"""

from ._buffer import Buffer
from ._stream import StreamMarker, StreamEEG
from .stream_receiver import StreamReceiver
