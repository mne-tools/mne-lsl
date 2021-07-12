import argparse

from pathlib import Path

from neurodecode.stream_recorder import StreamRecorder


def run():
    parser = argparse.ArgumentParser(prog='StreamRecorder')
    parser.add_argument(
        '-d', nargs='?', help='help for -d: Directory to save data.')
    parser.add_argument(
        '-f', nargs='?', help='help for -f: Filename stem.')
    parser.add_argument(
        '-s', nargs='?', help='help for -s: Stream name(s) to record.')

    args = parser.parse_args()
    record_dir = Path(args.d)
    fname = args.f
    stream_name = args.s

    recorder = StreamRecorder(record_dir, fname, stream_name)
    recorder.start(verbose=True)
    input(">> Press ENTER to stop the recording \n")
    recorder.stop()
