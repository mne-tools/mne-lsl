import pytest

from bsl.lsl.utils import _check_timeout


def test_check_timeout():
    """Test timeout static checker."""
    assert _check_timeout(0) == 0
    assert 10000 <= _check_timeout(None)
    assert _check_timeout(2) == 2
    assert _check_timeout(2.2) == 2.2

    with pytest.raises(
        TypeError, match="The argument 'timeout' must be a strictly positive"
    ):
        _check_timeout([2])
    with pytest.raises(
        ValueError, match="The argument 'timeout' must be a strictly positive"
    ):
        _check_timeout(-2.2)
