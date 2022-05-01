"""
Acquires signals from LSL server and save them to '-raw.fif' files in a
record directory.

Command-line arguments:
    -d --directory      Path to the record directory (str, path)
    -f --fname          File name stem (str)
    -s --stream_name    Stream name (str)
    --fif_subdir        Flag to save .fif in a subdirectory.
    --verbose           Flag to display a timer every recorded second.
If no argument is provided, records in the current directory.

Example:
    bsl_stream_recorder -d "D:/Data"
    bsl_stream_recorder -d "D:/Data" -f test
    bsl_stream_recorder -d "D:/Data" -f test -s openvibeSignals
    bsl_stream_recorder -d "D:/Data" -f test -s openvibeSignals --fif_subdir
    bsl_stream_recorder -d "D:/Data" -f test -s openvibeSignals --verbose
"""

import argparse

from pathlib import Path

from bsl import StreamRecorder


def run():
    """Entrypoint for bsl_stream_recorder usage."""
    parser = argparse.ArgumentParser(
        prog="BSL StreamRecorder",
        description="Starts recording data from stream(s) on LSL network.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        metavar="str",
        help="directory where the recorded data is saved.",
        default=Path.cwd(),
    )
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        metavar="str",
        help="file name stem used to create the recorded files.",
    )
    parser.add_argument(
        "-s",
        "--stream_name",
        type=str,
        metavar="str",
        help="stream(s) to record.",
    )
    parser.add_argument(
        "--fif_subdir",
        action="store_true",
        help="save .fif in a subdirectory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="display a timer every recorded second.",
    )

    args = parser.parse_args()

    recorder = StreamRecorder(
        args.directory,
        args.fname,
        args.stream_name,
        args.fif_subdir,
        verbose=args.verbose,
    )
    recorder.start()
    input(">> Press ENTER to stop the recording \n")
    recorder.stop()
