"""
Visualize LSL stream on the network. The signal is visualizd in real time with
spectral filtering and common average filtering option.

Command-line arguments:
    -s --stream_name    Stream name (str)

Example:
    bsl_stream_viewer
    bsl_stream_viewer -s StreamPlayer
"""

import argparse

from bsl import StreamViewer


def run():
    """Entrypoint for bsl_stream_viewer usage."""
    parser = argparse.ArgumentParser(
        prog='BSL StreamViewer',
        description='Starts a real-time viewer for a stream on LSL network.')
    parser.add_argument(
        '-s', '--stream_name', type=str, metavar='str',
        help='stream to display/plot.')

    args = parser.parse_args()
    stream_name = args.stream_name

    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start()
