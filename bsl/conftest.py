from pytest import fixture

from bsl import Player
from bsl.datasets import testing


@fixture(scope="module")
def mock_lsl_stream():
    """Create a mock LSL stream for testing."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    with Player(fname, "BSL-Player-pytest", chunk_size=16):
        yield
