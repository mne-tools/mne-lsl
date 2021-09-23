import time

import mne

from bsl import StreamRecorder, logger, set_log_level
from bsl.datasets import sample
from bsl.triggers.lpt import TriggerLPT, TriggerUSB2LPT, TriggerArduino2LPT
from bsl.utils._testing import (Stream, requires_sample_dataset, requires_lpt,
                                requires_usb2lpt, requires_arduino2lpt)


set_log_level('INFO')
logger.propagate = True

# TODO: Compact the syntax into one parametized function.


@requires_lpt
@requires_sample_dataset
def test_lpt(tmp_path, portaddr):
    """Testing for build-in LPT port."""
    # TODO
    pass


@requires_usb2lpt
@requires_sample_dataset
def test_usblpt(tmp_path, caplog):
    """Testing for USB to LPT converter."""
    # Test trigger
    with Stream('StreamPlayer', sample):
        recorder = StreamRecorder(record_dir=tmp_path, fname='test',
                                  stream_name='StreamPlayer')
        recorder.start(fif_subdir=False)

        trigger = TriggerUSB2LPT(verbose=True)
        assert trigger.verbose
        trigger.verbose = False

        assert trigger.signal(1)
        time.sleep(0.1)
        assert trigger.signal(2)

        recorder.stop()

    raw = mne.io.read_raw_fif(tmp_path / 'test-StreamPlayer-raw.fif')
    events = mne.find_events(raw, stim_channel='TRIGGER')
    assert events.shape == (2, 3)
    assert (events[:, 2] == [1, 2]).all()

    # Test delay
    trigger = TriggerUSB2LPT(delay=100, verbose=False)
    time.sleep(0.1)
    assert trigger.signal(1)
    assert not trigger.signal(2)
    assert 'new signal before the end of the last' in caplog.text

    # Test property setters
    time.sleep(0.2)
    trigger.delay = 50
    assert trigger.delay == 50
    assert trigger.signal(1)
    time.sleep(0.05)
    assert trigger.signal(2)
    trigger.delay = 1000
    assert trigger.delay == 50
    assert 'delay while an event' in caplog.text


@requires_arduino2lpt
@requires_sample_dataset
def test_arduino2lpt(tmp_path, caplog):
    """Testing for Arduino to LPT converter."""
    # Test trigger
    with Stream('StreamPlayer', sample):
        recorder = StreamRecorder(record_dir=tmp_path, fname='test',
                                  stream_name='StreamPlayer')
        recorder.start(fif_subdir=False)

        trigger = TriggerArduino2LPT(verbose=True)
        assert trigger.verbose
        trigger.verbose = False

        assert trigger.signal(1)
        time.sleep(0.1)
        assert trigger.signal(2)

        trigger.close()
        recorder.stop()

    raw = mne.io.read_raw_fif(tmp_path / 'test-StreamPlayer-raw.fif')
    events = mne.find_events(raw, stim_channel='TRIGGER')
    assert events.shape == (2, 3)
    assert (events[:, 2] == [1, 2]).all()

    # Test delay
    trigger = TriggerArduino2LPT(delay=100, verbose=False)
    time.sleep(0.1)
    assert trigger.signal(1)
    assert not trigger.signal(2)
    assert 'new signal before the end of the last' in caplog.text

    # Test property setters
    time.sleep(0.2)
    trigger.delay = 50
    assert trigger.delay == 50
    assert trigger.signal(1)
    time.sleep(0.05)
    assert trigger.signal(2)
    trigger.delay = 1000
    assert trigger.delay == 50
    assert 'delay while an event' in caplog.text
