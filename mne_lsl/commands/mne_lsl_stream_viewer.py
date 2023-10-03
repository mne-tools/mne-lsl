import argparse

from mne_lsl.stream_viewer import StreamViewer


def run():
    """Entrypoint for mne_lsl_stream_viewer usage."""
    parser = argparse.ArgumentParser(
        prog="MNE-LSL StreamViewer",
        description="Starts a real-time viewer for a stream on LSL network.",
    )
    parser.add_argument(
        "-s",
        "--stream_name",
        type=str,
        metavar="str",
        help="stream to display/plot.",
    )

    args = parser.parse_args()
    stream_name = args.stream_name

    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start()
