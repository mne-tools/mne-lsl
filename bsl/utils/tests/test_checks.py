"""Test _checks.py"""

import logging

import pytest

from bsl.utils._checks import (
    _check_type,
    _check_value,
    _check_verbose,
    _ensure_int,
)


def test_ensure_int():
    """Test _ensure_int checker."""
    # valids
    assert _ensure_int(101) == 101

    # invalids
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


def test_check_verbose():
    """Test _check_verbose checker."""
    # valids
    assert _check_verbose(12) == 12
    assert _check_verbose("INFO") == logging.INFO
    assert _check_verbose("DEBUG") == logging.DEBUG
    assert _check_verbose(True) == logging.INFO
    assert _check_verbose(False) == logging.WARNING
    assert _check_verbose(None) == logging.WARNING

    # invalids
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_verbose(("INFO",))
    with pytest.raises(ValueError, match="Invalid value"):
        _check_verbose("101")
    with pytest.raises(ValueError, match="negative integer, -101 is invalid."):
        _check_verbose(-101)
