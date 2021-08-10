"""
NeuroDecode provides a real-time brain signal decoding framework.
"""
from ._version import __version__
from .logger import logger, set_log_level
from .stream_receiver import StreamReceiver
from .stream_recorder import StreamRecorder
from .stream_player import StreamPlayer
from .stream_viewer import StreamViewer
