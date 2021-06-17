#!/usr/bin/env python3

"""
Visualize LSL stream on the network. The signal is visualizd in real time with
spectral filtering, common average filtering option and real-time FFT.

Command-line arguments:
    #1 Stream name

Example:
    python nd_stream_viewer.py
    python nd_stream_viewer.py StreamPlayer

If the scripts have been added to the PATH (c.f. github), can be called
from terminal with the command nd_stream_recorder.
Example:
    nd_stream_viewer
    nd_stream_viewer StreamPlayer
"""

from neurodecode.stream_viewer import StreamViewer

if __name__ == '__main__':

    import sys

    stream_name = None

    if len(sys.argv) > 2:
        raise RuntimeError("Too many arguments provided, maximum is 1.")

    if len(sys.argv) > 1:
        stream_name = sys.argv[1]

    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start()
