# (not using import * for Python 2.5 support)
from .pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    resolve_streams,
)
from .functions import library_version, protocol_version, local_clock
