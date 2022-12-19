# (not using import * for Python 2.5 support)
from .pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    library_version,
    local_clock,
    protocol_version,
    resolve_streams,
)
from .version import __version__
