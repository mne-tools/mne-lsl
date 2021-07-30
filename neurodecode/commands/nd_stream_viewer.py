"""
Visualize LSL stream on the network. The signal is visualizd in real time with
spectral filtering, common average filtering option and real-time FFT.

Command-line arguments:
    -s --stream_name    Stream name (str)
    -b --backend        Select plot backend (str).
                        Supported: pyqt5, vispy

Example:
    nd_stream_viewer
    nd_stream_viewer -s StreamPlayer
    nd_stream_viewer -s StreamPlayer -b vispy
"""

import argparse

from neurodecode.stream_viewer import StreamViewer


def run():
    parser = argparse.ArgumentParser(
        prog='StreamViewer',
        description='Starts a real-time viewer for a stream on LSL network.')
    parser.add_argument(
        '-s', '--stream_name', type=str, metavar='str',
        help='stream to display/plot.')
    parser.add_argument(
        '-b', '--backend', type=str, metavar='str',
        help='selected plot backend.', default='pyqt5')

    args = parser.parse_args()
    stream_name = args.stream_name
    backend = args.backend

    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start(backend=backend)


def main():
    """Entrypoint for nd_stream_viewer usage."""
    run()
