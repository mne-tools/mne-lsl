import pytest

from bsl.datasets._fetching import fetch_file
from bsl.utils._tests import requires_good_network


URL = 'https://raw.githubusercontent.com/bsl-tools/bsl/master/README.md'
HASH = {'md5': '077ea2b59000e405410b1ce50f0b33a8',
        'sha1': '29fbc8a62ed6a5325dec56e4d9e39589e2139f09'}
WRONG_HASH = {'md5': '12345678912345678912345678912345',
              'sha1': '1234567891234567891234567891234567891234'}


@requires_good_network
@pytest.mark.parametrize('hash_type', ('md5', 'sha1'))
def test_fetch_file(tmp_path, hash_type):
    """Test file download."""
    fetch_file(URL, tmp_path / 'README1.md',
               hash_=HASH[hash_type], hash_type=hash_type)
    # Test wrong hash value
    with pytest.raises(RuntimeError, match='Hash mismatch for downloaded'):
        fetch_file(URL, tmp_path / 'README2.md',
                   hash_=WRONG_HASH[hash_type], hash_type=hash_type)


def test_fetch_file_invalid_arg(tmp_path):
    """Test invalid URL and hash/hash types."""
    # Test invalid URL
    with pytest.raises(NotImplementedError, match='Cannot use scheme'):
        fetch_file('not an address', tmp_path / 'test.md')
    # Test invalid hash value
    with pytest.raises(ValueError, match='Bad hash value given'):
        fetch_file(URL, tmp_path / 'test.md', hash_='not an hash')
        fetch_file(URL, tmp_path / 'test.md', hash_=123456789)
    # Test invalid hash type
    with pytest.raises(ValueError, match='Unsupported hash type'):
        fetch_file(URL, tmp_path / 'README3.md',
                   hash_=HASH['md5'], hash_type='...')
        fetch_file(URL, tmp_path / 'README3.md',
                   hash_=HASH['md5'], hash_type=123)
