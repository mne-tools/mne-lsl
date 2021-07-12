import argparse

from neurodecode.stream_viewer import StreamViewer


def run():
    parser = argparse.ArgumentParser(prog='StreamViewer')
    parser.add_argument(
        '-n', nargs='?', help='help for -n: Name of the stream')

    args = parser.parse_args()
    stream_name = args.n

    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start()
