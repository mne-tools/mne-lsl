from bsl.datasets import sample
from bsl.datasets._fetching import _hashfunc
from bsl.utils._testing import requires_good_network


@requires_good_network
def test_data_path():
    """Test that the return path exist and match the desired dataset."""
    path = sample.data_path()
    hashsum = _hashfunc(path, hash_type='md5')
    assert path.exists()
    assert hashsum == sample.MD5
