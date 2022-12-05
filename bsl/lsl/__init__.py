"""
Low-level API for the Lab Streaming Layer (LSL).

The API is similar to `pylsl`_, but the code-base is restructured, improved,
and tested on the OS and version of Python supported by ``BSL``.

.. _pylsl: https://github.com/labstreaminglayer/liblsl-Python
"""

from .functions import library_version, local_clock  # noqa: F401
from .stream_info import StreamInfo  # noqa: F401
