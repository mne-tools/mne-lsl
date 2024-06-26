import argparse

from ..stream_viewer import StreamViewer
from ..utils.logs import warn


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
    warn(
        "The CLI entry-points 'mne_lsl_stream_viewer' and 'mne_lsl_player' are "
        "deprecated in favor of 'mne-lsl viewer' and 'mne-lsl player'. They will be "
        "removed in MNE-LSL 1.6.",
        FutureWarning,
    )
    args = parser.parse_args()
    stream_name = args.stream_name
    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start()
