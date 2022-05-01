"""
BSL provides a real-time brain signal streaming framework.
"""

from . import datasets
from ._version import __version__
from .stream_player import StreamPlayer
from .stream_receiver import StreamReceiver
from .stream_recorder import StreamRecorder
from .stream_viewer import StreamViewer
from .utils._logs import set_log_level
from .utils._logs import (
    add_file_handler,
    add_stream_handler,
    logger,
    set_handler_log_level,
)
