import argparse

from bsl.player import PlayerLSL


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
    parser.add_argument("type", type=str, help="type of mock stream (supported: lsl).")
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
        default="BSL-Player",
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
