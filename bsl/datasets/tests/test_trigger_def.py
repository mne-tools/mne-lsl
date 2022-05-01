from bsl.datasets import trigger_def
from bsl.datasets._fetching import _hashfunc
from bsl.utils._tests import requires_good_network


@requires_good_network
def test_data_path():
    """Test that the path exist and match the desired dataset."""
    path = trigger_def.data_path()
    assert path.exists()
    assert _hashfunc(path, hash_type="md5") == trigger_def.MD5
    assert _hashfunc(path, hash_type="sha1") == trigger_def.SHA1
