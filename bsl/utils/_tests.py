"""Utility functions for testing. Inspired from MNE."""

import time
import ctypes
import requests
from pathlib import Path
from functools import partial

import pytest

from ..datasets import (eeg_resting_state, eeg_resting_state_short,
                        eeg_auditory_stimuli, trigger_def)
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
        return requires_good_network(function)
    else:
        return function


requires_eeg_auditory_stimuli_dataset = partial(
    _requires_dataset_or_good_network, dataset=eeg_auditory_stimuli)
requires_eeg_resting_state_dataset = partial(
    _requires_dataset_or_good_network, dataset=eeg_resting_state)
requires_eeg_resting_state_short_dataset = partial(
    _requires_dataset_or_good_network, dataset=eeg_resting_state_short)
requires_trigger_def_dataset = partial(
    _requires_dataset_or_good_network, dataset=trigger_def)


def requires_lpt(function):
    """Decorator to skip a test if a build-in LPT port is not available."""
    ext = '32.dll' if ctypes.sizeof(ctypes.c_voidp) == 4 else '64.dll'
    dllname = 'LptControl_Desktop' + ext
    dll = Path(__file__).parent.parent / 'triggers' / 'lpt_libs' / dllname
    try:
        lpt = ctypes.cdll.LoadLibrary(str(dll))
    except Exception:
        return pytest.mark.skipif(True, reason='LPT dll not found.')(function)
    for portaddr in [0x278, 0x378]:
        try:
            lpt.setdata(portaddr, 1)
            return function
        except Exception:
            pass
    return pytest.mark.skipif(True, reason='LPT port not found.')(function)


def requires_usb2lpt(function):
    """Decorator to skip a test if a USB to LPT converter is not available."""
    ext = '32.dll' if ctypes.sizeof(ctypes.c_voidp) == 4 else '64.dll'
    dllname = 'LptControl_USB2LPT' + ext
    dll = Path(__file__).parent.parent / 'triggers' / 'lpt_libs' / dllname
    try:
        lpt = ctypes.cdll.LoadLibrary(str(dll))
    except Exception:
        return pytest.mark.skipif(True, reason='LPT dll not found.')(function)
    try:
        lpt.setdata(1)
        return function
    except Exception:
        return pytest.mark.skipif(True, reason='LPT port not found.')(function)


def requires_arduino2lpt(function):
    """Decorator to skip a test if an Arduino to LPT converter is not
    available."""
    try:
        import serial
        from serial.tools import list_ports
    except ModuleNotFoundError:
        return pytest.mark.skipif(
            True, reason='pyserial not installed.')(function)
    for arduino in list_ports.grep(regexp='Arduino'):
        try:
            ser = serial.Serial(arduino.device, 115200)
            time.sleep(0.2)
            ser.write(bytes([1]))
            ser.close()
            return function
        except Exception:
            pass
    return pytest.mark.skipif(
        True, reason='Arduino to LPT not found.')(function)
