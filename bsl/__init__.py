"""
BSL provides a real-time brain signal streaming framework.
"""

from ._version import __version__  # noqa: F401
from .stream_receiver import StreamReceiver  # noqa: F401
from .stream_recorder import StreamRecorder  # noqa: F401
from .stream_player import StreamPlayer  # noqa: F401
from .stream_viewer import StreamViewer  # noqa: F401
from . import datasets  # noqa: F401
from .utils._logs import (
    logger,
    set_log_level,  # noqa: F401
    set_handler_log_level,
    add_stream_handler,
    add_file_handler,
)
