#!/usr/bin/env python3

"""
Stream a recorded fif file on LSL network.

Command-line arguments:
    #1: Server name
    #2: Raw FIF file to stream
    #3: Chunk size
    #4: Trigger file
If no argument is supplied, prompts for the server name and for the path to
the '-raw.fif' file to play.
Example:
    python nd_stream_player.py StreamPlayer "D:/Data/sample-raw.fif"
    python nd_stream_player.py StreamPlayer "D:/Data/sample-raw.fif" 16

If the scripts have been added to the PATH (c.f. github), can be called
from terminal with the command nd_stream_recorder.
Example:
    nd_stream_player StreamPlayer "D:/Data/sample-raw.fif"
    nd_stream_recorder StreamPlayer "D:/Data/sample-raw.fif" 16
"""

from neurodecode.stream_player import StreamPlayer

if __name__ == '__main__':

    import sys
    from pathlib import Path

    chunk_size = 16
    trigger_file = None
    fif_file = None
    server_name = None

    if len(sys.argv) > 5:
        raise RuntimeError("Too many arguments provided, maximum is 4.")

    if len(sys.argv) > 4:
        trigger_file = sys.argv[4]

    if len(sys.argv) > 3:
        chunk_size = int(sys.argv[3])

    if len(sys.argv) > 2:
        fif_file = sys.argv[2]

    if len(sys.argv) > 1:
        server_name = sys.argv[1]
        if not fif_file:
            fif_file = str(
                Path(input(">> Provide the path to the .fif file to play: \n>> ")))

    if len(sys.argv) == 1:
        server_name = input(
            ">> Provide the server name displayed on LSL network: \n>> ")
        fif_file = str(
            Path(input(">> Provide the path to the .fif file to play: \n>> ")))

    sp = StreamPlayer(server_name, fif_file, chunk_size, trigger_file)
    sp.start()
    input(">> Press ENTER to stop replaying data \n")
    sp.stop()
