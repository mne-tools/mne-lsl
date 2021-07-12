import time
import argparse

from pathlib import Path

from neurodecode.stream_player import StreamPlayer


def run():
    parser = argparse.ArgumentParser(prog='StreamPlayer')
    parser.add_argument(
        '-n', nargs='?', help='help for -n: Name of the stream')
    parser.add_argument(
        '-f', nargs='?', help='help for -f: FIF File to stream')
    parser.add_argument(
        '-c', nargs='?', help='help for -c: Chunk_size')
    parser.add_argument(
        '-t', nargs='?', help='help for -w: Trigger file')


    args = parser.parse_args()
    server_name = args.n
    fif_file = Path(args.f)
    chunk_size = int(args.c) if args.c is not None else 16
    trigger_file = args.t

    if server_name is None:
        server_name = input(
            ">> Provide the server name displayed on LSL network: \n>> ")
    if fif_file is None:
        fif_file = str(
            Path(input(">> Provide the path to the .fif file to play: \n>> ")))

    sp = StreamPlayer(server_name, fif_file, chunk_size, trigger_file)
    sp.start()
    time.sleep(0.5)
    input(">> Press ENTER to stop replaying data \n")
    sp.stop()
