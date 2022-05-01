"""Test _checks.py"""

import pytest

from bsl.utils._checks import _ensure_int, _check_type, _check_value


def test_ensure_int():
    """Test _ensure_int checker."""
    assert _ensure_int(101) == 101
    with pytest.raises(TypeError, match="Item must be an int"):
        _ensure_int(101.0)
    with pytest.raises(TypeError, match="Item must be an int"):
        _ensure_int(True)
    with pytest.raises(TypeError, match="Item must be an int"):
        _ensure_int([101])


def test_check_type():
    """Test _check_type checker."""
    # valids
    assert _check_type(101, ("int",)) == 101
    assert _check_type("101.fif", ("path-like",)) == "101.fif"

    def foo():
        pass

    _check_type(foo, ("callable",))

    assert _check_type(101, ("numeric",)) == 101
    assert _check_type(101.0, ("numeric",)) == 101.0

    # invalids
    with pytest.raises(TypeError, match="Item must be an instance of"):
        _check_type(101, (float,))
    with pytest.raises(TypeError, match="'number' must be an instance of"):
        _check_type(101, (float,), "number")


def test_check_value():
    """Test _check_value checker."""
    # valids
    assert _check_value(5, [1, 2, 3, 4, 5]) == 5
    assert _check_value((1, 2), [(1, 2), (2, 3, 4, 5)]) == (1, 2)

    # invalids
    with pytest.raises(ValueError, match="Invalid value for the parameter."):
        _check_value(5, [1, 2, 3, 4])
    with pytest.raises(
        ValueError, match="Invalid value for the 'number' parameter."
    ):
        _check_value(5, [1, 2, 3, 4], "number")
