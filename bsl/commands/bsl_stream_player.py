"""
Stream a recorded .fif file on LSL network.

Command-line arguments:
    #1                  Stream name (str)
    #2                  Raw FIF file to stream (str, path)
    -c --chunk_size     Chunk size (int)
    -t --trigger_file   Trigger file (str, path)

the '-raw.fif' file to play.
Example:
    bsl_stream_player StreamPlayer "D:/Data/sample-raw.fif"
    bsl_stream_player StreamPlayer "D:/Data/sample-raw.fif" -c 16
    bsl_stream_player StreamPlayer "D:/Data/sample-raw.fif" -c 16 -t "D:/triggerdef_template.ini"
"""

import time
import argparse

from pathlib import Path

from bsl import StreamPlayer


def run():
    """Entrypoint for bsl_stream_player usage."""
    parser = argparse.ArgumentParser(
        prog='StreamPlayer',
        description='Starts streaming data from a .fif on the network.')
    parser.add_argument(
        'stream_name', type=str,
        help='name of the stream displayed on LSL network.')
    parser.add_argument(
        'fif_file', type=str,
        help='path to the FIF File to stream on LSL network.')
    parser.add_argument(
        '-c', '--chunk_size', type=int, metavar='int',
        help='chunk size, usually 16 or 32.', default=16)
    parser.add_argument(
        '-t', '--trigger_file', type=str, metavar='str',
        help='path to the trigger file.', default=None)

    args = parser.parse_args()

    server_name = args.stream_name
    fif_file = Path(args.fif_file)
    chunk_size = int(args.chunk_size)
    trigger_file = args.trigger_file

    sp = StreamPlayer(server_name, fif_file, chunk_size, trigger_file)
    sp.start()
    time.sleep(0.5)
    input(">> Press ENTER to stop replaying data \n")
    sp.stop()
