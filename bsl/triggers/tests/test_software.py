import time

import mne

from bsl import StreamRecorder, StreamPlayer
from bsl.datasets import eeg_resting_state
from bsl.triggers import SoftwareTrigger
from bsl.utils._tests import requires_eeg_resting_state_dataset


@requires_eeg_resting_state_dataset
def test_trigger_software(tmp_path):
    """Test software triggers."""
    with StreamPlayer('StreamPlayer', eeg_resting_state.data_path()):
        recorder = StreamRecorder(record_dir=tmp_path, fname='test',
                                  stream_name='StreamPlayer', fif_subdir=False)
        recorder.start()

        trigger = SoftwareTrigger(recorder, verbose=True)
        assert trigger.verbose
        trigger.verbose = False
        time.sleep(0.5)

        assert trigger.signal(1)
        time.sleep(0.1)
        assert trigger.signal(2)

        time.sleep(0.5)
        trigger.close()
        time.sleep(0.5)
        recorder.stop()

    raw = mne.io.read_raw_fif(tmp_path / 'test-StreamPlayer-raw.fif')
    events = mne.find_events(raw, stim_channel='TRIGGER')
    assert events.shape == (2, 3)
    assert (events[:, 2] == [1, 2]).all()
