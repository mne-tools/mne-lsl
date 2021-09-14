"""Utility function for testing. Inspired from MNE."""

import requests
from pathlib import Path
from functools import partial

import pytest

from .. import StreamPlayer
from ..datasets import sample
from ..datasets._fetching import _hashfunc


def requires_good_network(function):
    """Decorator to skip a test if a network connection is not available."""
    try:
        requests.get('https://github.com/', timeout=1)
        skip = False
    except ConnectionError:
        skip = True
    name = function.__name__
    reason = f'Test {name} skipped, requires a good network connection.'
    return pytest.mark.skipif(skip, reason=reason)(function)


def _requires_dataset_or_good_network(function, dataset):
    """Decorator to skip a test if a required dataset is absent and it can not
    be downloaded."""
    # BSL datasets
    try:
        fname = dataset.PATH
        download = False if fname.exists() \
            and _hashfunc(fname, hash_type='md5') == dataset.MD5 else True
    except AttributeError:
        # MNE datasets
        try:
            fname = dataset.data_path(download=False)
            download = False if fname != '' and Path(fname).exists() else True
        except AttributeError:
            raise ValueError('Unsupported dataset.')

    if download:
        requires_good_network(function)
    else:
        return function


requires_sample_dataset = partial(_requires_dataset_or_good_network,
                                  dataset=sample)


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
