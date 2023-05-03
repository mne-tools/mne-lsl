"""Utility functions for testing. Inspired from MNE."""

from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Callable

import pytest
import requests

from ..datasets import (
    eeg_auditory_stimuli,
    eeg_resting_state,
    eeg_resting_state_short,
    trigger_def,
)
from ..datasets._fetching import _hashfunc


def requires_good_network(function: Callable):  # noqa: D401
    """Decorator to skip a test if a network connection is not available."""
    try:
        requests.get("https://github.com/", timeout=1)
        skip = False
    except ConnectionError:
        skip = True
    name = function.__name__
    reason = f"Test {name} skipped, requires a good network connection."
    return pytest.mark.skipif(skip, reason=reason)(function)


def _requires_dataset_or_good_network(function: Callable, dataset):  # noqa
    """Decorator to skip a test if a required dataset is absent and it can not
    be downloaded.
    """
    # BSL datasets
    try:
        fname = dataset.PATH
        download = (
            False
            if fname.exists() and _hashfunc(fname, hash_type="md5") == dataset.MD5
            else True
        )
    except AttributeError:
        # MNE datasets
        try:
            fname = dataset.data_path(download=False)
            download = False if fname != "" and Path(fname).exists() else True
        except AttributeError:
            raise ValueError("Unsupported dataset.")

    if download:
        return requires_good_network(function)
    else:
        return function


requires_eeg_auditory_stimuli_dataset = partial(
    _requires_dataset_or_good_network, dataset=eeg_auditory_stimuli
)
requires_eeg_resting_state_dataset = partial(
    _requires_dataset_or_good_network, dataset=eeg_resting_state
)
requires_eeg_resting_state_short_dataset = partial(
    _requires_dataset_or_good_network, dataset=eeg_resting_state_short
)
requires_trigger_def_dataset = partial(
    _requires_dataset_or_good_network, dataset=trigger_def
)


def _requires_module(function: Callable, name: str):
    """Skip a test if package is not available (decorator)."""
    try:
        import_module(name)
        skip = False
    except ImportError:
        skip = True
    reason = f"Test {function.__name__} skipped, requires {name}."
    return pytest.mark.skipif(skip, reason=reason)(function)
