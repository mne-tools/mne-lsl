"""Utility function for testing. Inspired from MNE."""

import time
import ctypes
import requests
from pathlib import Path
from functools import partial

import pytest

from .. import StreamPlayer
from ..datasets import sample, event
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
requires_event_dataset = partial(_requires_dataset_or_good_network,
                                 dataset=event)


def requires_lpt(function):
    """Decorator to skip a test if a build-in LPT port is not available."""
    ext = '32.dll' if ctypes.sizeof(ctypes.c_voidp) == 4 else '64.dll'
    dllname = 'LptControl_Desktop' + ext
    dll = Path(__file__).parent.parent / 'triggers' / 'lpt_libs' / dllname
    try:
        lpt = ctypes.cdll.LoadLibrary(str(dll))
    except Exception:
        return pytest.mark.skipif(True, reason='LPT dll not found.')(function)
    lpt = ctypes.cdll.LoadLibrary(str(dll))
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
        pass
    return pytest.mark.skipif(True, reason='LPT port not found.')(function)


def requires_arduino2lpt(function):
    """Decorator to skip a test if an Arduino to LPT converter is not
    available."""
    import serial
    from serial.tools import list_ports
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


class Stream:
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
        time.sleep(2)  # wait for stream player to start

    def __exit__(self, *args):
        self.sp.stop()
        del self.sp
