from bsl.datasets import eeg_auditory_stimuli
from bsl.datasets._fetching import _hashfunc
from bsl.utils._tests import requires_good_network


@requires_good_network
def test_data_path():
    """Test that the path exist and match the desired dataset."""
    path = eeg_auditory_stimuli.data_path()
    assert path.exists()
    assert _hashfunc(path, hash_type='md5') == eeg_auditory_stimuli.MD5
    assert _hashfunc(path, hash_type='sha1') == eeg_auditory_stimuli.SHA1
