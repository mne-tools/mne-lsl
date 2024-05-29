import argparse

import numpy as np

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
        default=10,
    )
    parser.add_argument(
        "-n_repeat",
        "--number_repeat",
        type=int,
        metavar="int",
        help="number of times to repeat the file. By default, infinity.",
        default=np.inf,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="str",
        help="name of the stream displayed by LSL.",
        default="MNE-LSL-Player",
    )
    parser.add_argument(
        "--annotations",
        help="enable streaming of annotations",
        action="store_true",
    )
    args = parser.parse_args()
    player = PlayerLSL(
        fname=args.fname,
        chunk_size=args.chunk_size,
        n_repeat=args.number_repeat,
        name=args.name,
        annotations=args.annotations,
    )
    player.start()
    input(">> Press ENTER to stop replaying data \n")
    player.stop()
