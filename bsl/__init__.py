"""
BSL provides a real-time brain signal streaming framework.
"""

from . import datasets  # noqa: F401
from ._version import __version__  # noqa: F401
from .stream_player import StreamPlayer  # noqa: F401
from .stream_receiver import StreamReceiver  # noqa: F401
from .stream_recorder import StreamRecorder  # noqa: F401
from .stream_viewer import StreamViewer  # noqa: F401
from .utils._logs import (  # noqa: F401
    add_file_handler,
    add_stream_handler,
    logger,
    set_handler_log_level,
    set_log_level,
)
