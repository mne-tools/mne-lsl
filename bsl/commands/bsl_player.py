import argparse

from bsl import Player


def run():
    """Entrypoint for bsl_player usage."""
    parser = argparse.ArgumentParser(
        prog="BSL Player",
        description="Starts streaming data from an MNE-compatible file on the network.",
    )
    parser.add_argument(
        "fname",
        type=str,
        help="path to the File to stream via LSL.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="name of the stream displayed by LSL.",
        default="BSL-Player",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        metavar="int",
        help="number of samples pushed at once via LSL.",
        default=16,
    )

    args = parser.parse_args()

    player = Player(args.fname, args.name, args.chunk_size)
    player.start()
    input(">> Press ENTER to stop replaying data \n")
    player.stop()
