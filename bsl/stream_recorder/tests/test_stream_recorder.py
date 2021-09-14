import time

import pytest

from bsl import StreamRecorder
from bsl.datasets import sample
from bsl.utils._testing import TestStream, requires_sample_dataset


@requires_sample_dataset
def test_recording(tmp_path):
    """Test recording capability of the stream recorder."""
    # Test with one stream
    with TestStream('StreamPlayer', sample):
        recorder = StreamRecorder(record_dir=tmp_path)
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        time.sleep(3)
        recorder.stop()
