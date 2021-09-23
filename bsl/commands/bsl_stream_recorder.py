"""
Acquires signals from LSL server and save them to '-raw.fif' files in a
record directory.

Command-line arguments:
    -d --directory      Path to the record directory (str, path)
    -f --fname          File name stem (str)
    -s --stream_name    Stream name (str)
If no argument is provided, records in the current directory.

Example:
    bsl_stream_recorder -d "D:/Data"
    bsl_stream_recorder -d "D:/Data" -f test
    bsl_stream_recorder -d "D:/Data" -f test -s openvibeSignals
"""

import argparse

from pathlib import Path

from bsl import StreamRecorder


def run():
    """Entrypoint for bsl_stream_recorder usage."""
    parser = argparse.ArgumentParser(
        prog='StreamRecorder',
        description='Starts recording data from stream(s) on LSL network.')
    parser.add_argument(
        '-d', '--directory', type=str, metavar='str',
        help='directory where the recorded data is saved.', default=Path.cwd())
    parser.add_argument(
        '-f', '--fname', type=str, metavar='str',
        help='file name stem used to create the recorded files.')
    parser.add_argument(
        '-s', '--stream_name', type=str, metavar='str',
        help='stream(s) to record.')

    args = parser.parse_args()

    record_dir = args.directory
    fname = args.fname
    stream_name = args.stream_name

    recorder = StreamRecorder(record_dir, fname, stream_name)
    recorder.start(verbose=True)
    input(">> Press ENTER to stop the recording \n")
    recorder.stop()
