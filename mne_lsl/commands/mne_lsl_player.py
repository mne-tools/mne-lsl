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
        "-t",
        "--type",
        type=str,
        help="type of mock stream (supported: lsl).",
        default="lsl",
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
        "-c",
        "--chunk_size",
        type=int,
        metavar="int",
        help="number of samples pushed at once via LSL.",
        default=16,
    )

    args = parser.parse_args()

    if args.type.lower().strip() == "lsl":
        player = PlayerLSL(args.fname, args.name, args.chunk_size)
    else:
        raise ValueError(
            "Argument 'type' could not be interpreted as a known player. "
            f"Supported values are (lsl,). '{args.type}' is invalid."
        )

    player.start()
    input(">> Press ENTER to stop replaying data \n")
    player.stop()
