#!/usr/bin/env python3

"""
Acquires signals from LSL server and save them to '-raw.fif' files in a
record directory.

Command-line arguments:
    #1: Path to the record directory
    #2: Stream name (optional)
If no argument is supplied, prompts for a path to a record directory.
Example:
    python nd_stream_recorder.py "D:/Data"
    python nd_stream_recorder.py "D:/Data" openvibeSignals

If the scripts have been added to the PATH (c.f. github), can be called
from terminal with the command nd_stream_recorder.
Example:
    nd_stream_recorder "D:/Data"
    nd_stream_recorder "D:/Data" openvibeSignals
"""

from neurodecode.stream_recorder import StreamRecorder

if __name__ == '__main__':

    import sys
    from pathlib import Path

    stream_name = None

    if len(sys.argv) > 3:
        raise RuntimeError("Too many arguments provided, maximum is 2.")

    if len(sys.argv) == 3:
        record_dir = sys.argv[1]
        stream_name = sys.argv[2]

    if len(sys.argv) == 2:
        record_dir = sys.argv[1]

    if len(sys.argv) == 1:
        record_dir = str(
            Path(input(">> Provide the path to save the .fif file: \n>> ")))

    recorder = StreamRecorder(record_dir)
    recorder.start(stream_name=stream_name, eeg_only=False, verbose=True)
    input(">> Press ENTER to stop the recording \n")
    recorder.stop()
