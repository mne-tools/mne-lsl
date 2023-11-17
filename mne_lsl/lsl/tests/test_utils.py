from ctypes import c_int

import pytest

from mne_lsl.lsl._utils import (
    InternalError,
    InvalidArgumentError,
    LostError,
    check_timeout,
    handle_error,
)


def test_check_timeout():
    """Test timeout static checker."""
    assert check_timeout(0) == 0
    assert 10000 <= check_timeout(None)
    assert check_timeout(2) == 2
    assert check_timeout(2.2) == 2.2

    with pytest.raises(
        TypeError, match="The argument 'timeout' must be a strictly positive"
    ):
        check_timeout([2])
    with pytest.raises(
        ValueError, match="The argument 'timeout' must be a strictly positive"
    ):
        check_timeout(-2.2)


def test_handle_error():
    """Test error-code handler."""
    handle_error(0)
    handle_error(c_int(0))
    with pytest.raises(TimeoutError, match="due to a timeout."):
        handle_error(-1)
    with pytest.raises(LostError, match="connection has been lost."):
        handle_error(-2)
    with pytest.raises(
        InvalidArgumentError, match="argument was incorrectly specified."
    ):
        handle_error(-3)
    with pytest.raises(InternalError, match="internal error has occurred."):
        handle_error(-4)
    with pytest.raises(RuntimeError, match="unknown error"):
        handle_error(-101)
