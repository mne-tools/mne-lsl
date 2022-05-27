import time

import mne
import pytest

from bsl import StreamPlayer, StreamRecorder, logger, set_log_level
from bsl.datasets import eeg_resting_state
from bsl.triggers import ParallelPortTrigger
from bsl.utils._tests import (
    requires_arduino2lpt,
    requires_eeg_resting_state_dataset,
    requires_parallel,
)

set_log_level("INFO")
logger.propagate = True

# TODO: Compact the syntax into one parametized function.


@requires_parallel
@requires_eeg_resting_state_dataset
def test_parallel(tmp_path, portaddr):
    """Testing for built-in parallel port."""
    # TODO
    pass


@requires_arduino2lpt
@requires_eeg_resting_state_dataset
def test_parallel_with_arduino(tmp_path, caplog):
    """Testing for Arduino to LPT converter."""
    # Test trigger
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        recorder = StreamRecorder(
            record_dir=tmp_path,
            fname="test",
            stream_name="StreamPlayer",
            fif_subdir=False,
        )

        trigger = ParallelPortTrigger(address="arduino", verbose=True)
        time.sleep(0.5)
        assert "using an Arduino converter." in caplog.text
        assert trigger.verbose
        trigger.verbose = False

        assert trigger.signal(1)
        time.sleep(0.1)
        assert trigger.signal(2)

        trigger.close()
        time.sleep(0.5)
        recorder.stop()

    raw = mne.io.read_raw_fif(tmp_path / "test-StreamPlayer-raw.fif")
    events = mne.find_events(raw, stim_channel="TRIGGER")
    assert events.shape == (2, 3)
    assert (events[:, 2] == [1, 2]).all()

    # Test delay
    trigger = ParallelPortTrigger(address="arduino", delay=100, verbose=False)
    time.sleep(0.1)
    assert trigger.signal(1)
    assert not trigger.signal(2)
    assert "new signal before the end of the last" in caplog.text

    # Test property setters
    time.sleep(0.2)
    with pytest.raises(AttributeError, match="can't set attribute"):
        trigger.delay = 50
    assert trigger.delay == 100
