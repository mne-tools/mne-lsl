import time

import mne

from bsl import StreamRecorder
from bsl.datasets import sample
from bsl.triggers.software import TriggerSoftware
from bsl.utils._testing import Stream, requires_sample_dataset


@requires_sample_dataset
def test_trigger_software(tmp_path):
    """Test software triggers."""
    with Stream('StreamPlayer', sample):
        recorder = StreamRecorder(record_dir=tmp_path, fname='test',
                                  stream_name='StreamPlayer')
        recorder.start(fif_subdir=False)

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
