import time

import mne

from bsl import StreamRecorder, StreamPlayer
from bsl.datasets import eeg_resting_state
from bsl.triggers.software import TriggerSoftware
from bsl.utils._testing import requires_eeg_resting_state_dataset


@requires_eeg_resting_state_dataset
def test_trigger_software(tmp_path):
    """Test software triggers."""
    with StreamPlayer('StreamPlayer', eeg_resting_state.data_path()):
        recorder = StreamRecorder(record_dir=tmp_path, fname='test',
                                  stream_name='StreamPlayer', fif_subdir=False)
        recorder.start()

        trigger = TriggerSoftware(recorder, verbose=True)
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
