"""
Stream a recorded .fif file on LSL network.

Command-line arguments:
    #1                  Stream name (str)
    #2                  Raw FIF file to stream (str, path)
    -r --repeat         Number of repeat (int), default inf.
    -t --trigger_def    TriggerDef compatible file (str, path)
    -c --chunk_size     Chunk size (int)
    --high_resolution   Flag to use high resolution counter instead of sleep.

Examples:
    bsl_stream_player StreamPlayer "D:/Data/sample-raw.fif"
    bsl_stream_player StreamPlayer "D:/Data/sample-raw.fif" -r 5
    bsl_stream_player StreamPlayer "D:/Data/sample-raw.fif" -r 5 -c 16
    bsl_stream_player StreamPlayer "D:/Data/sample-raw.fif" --high_resolution
"""

import argparse

from bsl import StreamPlayer


def run():
    """Entrypoint for bsl_stream_player usage."""
    parser = argparse.ArgumentParser(
        prog='BSL StreamPlayer',
        description='Starts streaming data from a .fif on the network.')
    parser.add_argument(
        'stream_name', type=str,
        help='name of the stream displayed on LSL network.')
    parser.add_argument(
        'fif_file', type=str,
        help='path to the FIF File to stream on LSL network.')
    parser.add_argument(
        '-r', '--repeat', type=int, metavar='int',
        help='number of time the fif file is repeated.', default=float('inf'))
    parser.add_argument(
        '-t', '--trigger_def', type=str, metavar='str',
        help='path to a TriggerDef compatible file.', default=None)
    parser.add_argument(
        '-c', '--chunk_size', type=int, metavar='int',
        help='chunk size, usually 16 or 32.', default=16)
    parser.add_argument(
        '--high_resolution', action='store_true',
        help='use time.perf_counter() instead of time.sleep().')

    args = parser.parse_args()

    sp = StreamPlayer(args.stream_name, args.fif_file, args.repeat,
                      args.trigger_def, args.chunk_size, args.high_resolution)
    sp.start()
    input(">> Press ENTER to stop replaying data \n")
    sp.stop()
