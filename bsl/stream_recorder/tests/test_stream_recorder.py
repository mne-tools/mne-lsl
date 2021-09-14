import time
import pickle

import mne

from bsl import StreamRecorder
from bsl.datasets import sample
from bsl.utils._testing import Stream, requires_sample_dataset


@requires_sample_dataset
def test_recording(tmp_path):
    """Test recording capability of the stream recorder."""
    # Test with one stream
    stream = 'StreamPlayer'
    duration = 3  # seconds
    with Stream(stream, sample):
        recorder = StreamRecorder(record_dir=tmp_path)
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(duration)
        recorder.stop()

    # Use the eve_file to retrieve the file name stem used
    fname_stem = eve_file.stem.split('-eve')[0]
    fname_pcl = tmp_path / f'{fname_stem}-{stream}-raw.pcl'
    fname_fif = tmp_path / f'{fname_stem}-{stream}-raw.fif'
    assert fname_pcl.exists()
    assert fname_fif.exists()

    # Check file content
    raw = mne.io.read_raw_fif(fname=sample.data_path(), preload=True)
    with open(fname_pcl, 'rb') as inp:
        raw_pcl = pickle.load(inp)
    raw_fif = mne.io.read_raw_fif(fname_fif, preload=True)
    assert raw.ch_names == raw_pcl['ch_names'] == raw_fif.ch_names
    assert raw.info['sfreq'] == raw_pcl['sample_rate'] == raw_fif.info['sfreq']
    assert raw_pcl['signals'].shape[::-1] == raw_fif.get_data().shape
    assert abs(raw_fif.n_times / raw_fif.info['sfreq'] - duration) < 0.2
