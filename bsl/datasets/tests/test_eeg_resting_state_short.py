from bsl.datasets import eeg_resting_state_short
from bsl.datasets._fetching import _hashfunc


def test_data_path():
    """Test that the path exist and match the desired dataset."""
    path = eeg_resting_state_short.data_path()
    assert path.exists()
    assert _hashfunc(path, hash_type="md5") == eeg_resting_state_short.MD5
    assert _hashfunc(path, hash_type="sha1") == eeg_resting_state_short.SHA1
