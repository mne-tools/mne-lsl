"""
.. include:: ../links.inc

Low-level python API for the Lab Streaming Layer (LSL).

The API is similar to `pylsl <lsl python_>`_, but the code-base is restructured,
improved, and tested on the OS and version of Python supported by ``BSL``.
"""

from .functions import (  # noqa: F401
    library_version,
    local_clock,
    protocol_version,
    resolve_streams,
)
from .stream_info import StreamInfo  # noqa: F401
from .stream_inlet import StreamInlet  # noqa: F401
from .stream_outlet import StreamOutlet  # noqa: F401
