import argparse

from pathlib import Path

from neurodecode.stream_recorder import StreamRecorder


def run():
    parser = argparse.ArgumentParser(
        prog='StreamRecorder',
        description='Starts recording data from stream(s) on LSL network.')
    parser.add_argument(
        '-d', '--directory', type=str, metavar='str',
        help='directory where the recorded data is saved.', default=Path.cwd())
    parser.add_argument(
        '-f', '--filename', type=str, metavar='str',
        help='filename stem used to create the recorded files.')
    parser.add_argument(
        '-s', '--stream_name', type=str, metavar='str',
        help='stream(s) to record.')

    args = parser.parse_args()

    record_dir = args.directory
    fname = args.filename
    stream_name = args.stream_name

    recorder = StreamRecorder(record_dir, fname, stream_name)
    recorder.start(verbose=True)
    input(">> Press ENTER to stop the recording \n")
    recorder.stop()


def main():
    """Entrypoint for nd_stream_recorder usage."""
    run()
