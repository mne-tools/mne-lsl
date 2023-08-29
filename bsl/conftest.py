from pytest import fixture

from bsl import Player
from bsl.datasets import eeg_resting_state_short


@fixture(scope="module")
def mock_lsl_stream():
    """Create a mock LSL stream for testing."""
    fname = eeg_resting_state_short.data_path()
    with Player(fname, "BSL-Player-pytest", chunk_size=16):
        yield
