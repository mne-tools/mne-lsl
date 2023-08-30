from bsl.datasets import eeg_resting_state
from bsl.datasets._fetching import _hashfunc


def test_data_path():
    """Test that the path exist and match the desired dataset."""
    path = eeg_resting_state.data_path()
    assert path.exists()
    assert _hashfunc(path, hash_type="md5") == eeg_resting_state.MD5
    assert _hashfunc(path, hash_type="sha1") == eeg_resting_state.SHA1
