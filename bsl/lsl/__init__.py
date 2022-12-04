"""
Low-level API for the Lab Streaming Layer (LSL).

The API is similar to `pylsl`_, but the code-base is restructured, improved,
and tested on the OS and version of Python supported by ``BSL``.

.. _pylsl: https://github.com/labstreaminglayer/liblsl-Python
"""

# Minimum liblsl version. The major version is given by version // 100 and the
# minor version is given by version % 100.
minversion = 115
