"""
Low-level API for the Lab Streaming Layer (LSL).

The API is similar to `pylsl`_, but the code-base is restructured, improved,
and tested on the OS and version of Python supported by ``BSL``.

.. _pylsl: https://github.com/labstreaminglayer/liblsl-Python
"""

from .utils import load_liblsl

lib = load_liblsl()
