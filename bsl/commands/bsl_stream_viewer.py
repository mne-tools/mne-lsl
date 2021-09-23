"""
Visualize LSL stream on the network. The signal is visualizd in real time with
spectral filtering and common average filtering option.

Command-line arguments:
    -s --stream_name    Stream name (str)
    -b --backend        Select plot backend (str).
                        Supported: pyqtgraph, vispy

Example:
    bsl_stream_viewer
    bsl_stream_viewer -s StreamPlayer
    bsl_stream_viewer -s StreamPlayer -b vispy
"""

import argparse

from bsl import StreamViewer


def run():
    """Entrypoint for bsl_stream_viewer usage."""
    parser = argparse.ArgumentParser(
        prog='StreamViewer',
        description='Starts a real-time viewer for a stream on LSL network.')
    parser.add_argument(
        '-s', '--stream_name', type=str, metavar='str',
        help='stream to display/plot.')
    parser.add_argument(
        '-b', '--backend', type=str, metavar='str',
        help='selected plot backend.', default='pyqtgraph')

    args = parser.parse_args()
    stream_name = args.stream_name
    backend = args.backend

    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start(backend=backend)
