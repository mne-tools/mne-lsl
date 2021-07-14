import argparse

from neurodecode.stream_viewer import StreamViewer


def run():
    parser = argparse.ArgumentParser(
        prog='StreamViewer',
        description='Starts a real-time viewer for a stream on LSL network.')
    parser.add_argument(
        '-s', '--stream_name', type=str, metavar='str',
        help='stream to display/plot.')

    args = parser.parse_args()
    stream_name = args.stream_name

    stream_viewer = StreamViewer(stream_name)
    stream_viewer.start()


def main():
    """Entrypoint for nd_stream_viewer usage."""
    run()
