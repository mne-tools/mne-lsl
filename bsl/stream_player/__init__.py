"""
Module for signal replay. Replay the recorded signals in real time as if it was
transmitted from a real acquisition server.

For Windows users, make sure to use the provided time resolution
tweak tool (TimerTool) to set to 500us the time resolution of the OS.
"""

from .stream_player import StreamPlayer  # noqa: F401

__all__ = ["StreamPlayer"]
