import numpy as np
from mne import create_info
from mne.io import RawArray
from pytest import fixture

from bsl.datasets import testing
from bsl.player import PlayerLSL


@fixture(scope="module")
def mock_lsl_stream():
    """Create a mock LSL stream for testing."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    with PlayerLSL(fname, "BSL-Player-pytest", chunk_size=16):
        yield


@fixture(scope="module")
def mock_lsl_stream_int(tmp_path_factory):
    """Create a mock LSL stream streaming the channel number continuously."""
    info = create_info(5, 1000, "eeg")
    data = np.full((5, 1000), np.arange(5).reshape(-1, 1))
    raw = RawArray(data, info)
    fname = tmp_path_factory.mktemp("data") / "int-raw.fif"
    raw.save(fname)
    with PlayerLSL(fname, "BSL-Player-integers-pytest", chunk_size=16):
        yield
