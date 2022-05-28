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
