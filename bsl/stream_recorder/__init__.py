"""
Module for signal recording.

Record signals into fif format, a standard format mainly used in MNE EEG
analysis library.
"""

from .stream_recorder import StreamRecorder  # noqa: F401


__all__ = ["StreamRecorder"]
