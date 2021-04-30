'''
Module for signal replay

Replay the recorded signals in real time as if it 
was transmitted from a real acquisition server.

For Windows users, make sure to use the provided time resolution
tweak tool to set to 500us time resolution of the OS.

'''

from .stream_player import StreamPlayer, Streamer