import argparse

from mne_lsl.player import PlayerLSL


def run():
    """Entrypoint for mne_lsl_player usage."""
    parser = argparse.ArgumentParser(
        prog="MNE-LSL Player",
        description="Starts streaming data from an MNE-compatible file on the network.",
    )
    parser.add_argument(
        "fname",
        type=str,
        help="path to the File to stream via LSL.",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        metavar="int",
        help="number of samples pushed at once via LSL.",
        default=16,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="str",
        help="name of the stream displayed by LSL.",
        default="MNE-LSL-Player",
    )
    args = parser.parse_args()
    player = PlayerLSL(args.fname, args.chunk_size, args.name)
    player.start()
    input(">> Press ENTER to stop replaying data \n")
    player.stop()
