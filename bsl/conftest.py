from pytest import fixture

from bsl.datasets import testing
from bsl.player import PlayerLSL


@fixture(scope="module")
def mock_lsl_stream():
    """Create a mock LSL stream for testing."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    with PlayerLSL(fname, "BSL-Player-pytest", chunk_size=16):
        yield
