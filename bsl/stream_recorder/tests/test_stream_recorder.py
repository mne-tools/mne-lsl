import time
from pathlib import Path

import pytest

from bsl import StreamRecorder
from bsl.datasets import sample
from bsl.utils._testing import Stream, requires_sample_dataset


@requires_sample_dataset
def test_recording(tmp_path):
    """Test recording capability of the stream recorder."""
    # Test with one stream
    stream = 'StreamPlayer'
    with Stream(stream, sample):
        recorder = StreamRecorder(record_dir=tmp_path)
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(3)
        recorder.stop()
    # Use the eve_file to retrieve the file name stem used
    fname_stem = eve_file.stem.split('-eve')[0]
    fname_pcl = tmp_path / f'{fname_stem}-{stream}-raw.pcl'
    fname_fif = tmp_path / f'{fname_stem}-{stream}-raw.pcl'
    assert fname_pcl.exists()
    assert fname_fif.exists()
