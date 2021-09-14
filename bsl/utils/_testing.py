"""Utility function for testing. Inspired from MNE."""

import requests

import pytest

from .. import StreamPlayer


def requires_good_network(function):
    """Decorator to skip a test if a network connection is not available."""
    try:
        requests.get('https://github.com/', timeout=1)
        skip = False
    except ConnectionError:
        skip = True
    name = function.__name__
    reason = 'Test %s skipped, requires a good network connection.' % name
    return pytest.mark.skipif(skip, reason=reason)(function)


class TestStream:
    """Context manager to create a test stream.
    dataset must have a `.data_path()` method. Compatible with MNE."""

    def __init__(self, stream_name, dataset, chunk_size=16, trigger_file=None,
                 repeat=float('inf'), high_resolution=False):
        self.stream_name = stream_name
        self.fif_file = dataset.data_path()
        self.chunk_size = chunk_size
        self.trigger_file = trigger_file
        self.repeat = repeat
        self.high_resolution = high_resolution

    def __enter__(self):
        self.sp = StreamPlayer(self.stream_name, self.fif_file,
                               self.chunk_size, self.trigger_file)
        self.sp.start(repeat=self.repeat, high_resolution=self.high_resolution)

    def __exit__(self, *args):
        self.sp.stop()
