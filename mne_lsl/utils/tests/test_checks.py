import logging
from pathlib import Path

import numpy as np
import pytest

from mne_lsl.utils._checks import (
    check_type,
    check_value,
    check_verbose,
    ensure_int,
    ensure_path,
)


def test_ensure_int():
    """Test ensure_int checker."""
    # valids
    assert ensure_int(101) == 101

    # invalids
    with pytest.raises(TypeError, match="Item must be an int"):
        ensure_int(101.0)
    with pytest.raises(TypeError, match="Item must be an int"):
        ensure_int(True)
    with pytest.raises(TypeError, match="Item must be an int"):
        ensure_int([101])


def test_check_type():
    """Test check_type checker."""
    # valids
    check_type(101, ("int-like",))
    check_type(101, ("int-like", str))
    check_type("101.fif", ("path-like",))

    def foo():
        pass

    check_type(foo, ("callable",))

    check_type(101, ("numeric",))
    check_type(101.0, ("numeric",))
    check_type((1, 0, 1), ("array-like",))
    check_type([1, 0, 1], ("array-like",))
    check_type(np.array([1, 0, 1]), ("array-like",))

    # invalids
    with pytest.raises(TypeError, match="Item must be an instance of"):
        check_type(101, (float,))
    with pytest.raises(TypeError, match="Item must be an instance of"):
        check_type(101, ("array-like",))
    with pytest.raises(TypeError, match="'number' must be an instance of"):
        check_type(101, (float,), "number")


def test_check_value():
    """Test check_value checker."""
    # valids
    check_value(5, (5,))
    check_value(5, (5, 101))
    check_value(5, [1, 2, 3, 4, 5])
    check_value((1, 2), [(1, 2), (2, 3, 4, 5)])

    # invalids
    with pytest.raises(ValueError, match="Invalid value for the parameter."):
        check_value(5, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="Invalid value for the 'number' parameter."):
        check_value(5, [1, 2, 3, 4], "number")
    with pytest.raises(ValueError, match="only allowed value is"):
        check_value(5, (1,))
    with pytest.raises(ValueError, match="Allowed values are 3 and 4"):
        check_value(5, (3, 4))


def test_check_verbose():
    """Test check_verbose checker."""
    # valids
    assert check_verbose(12) == 12
    assert check_verbose("INFO") == logging.INFO
    assert check_verbose("DEBUG") == logging.DEBUG
    assert check_verbose(True) == logging.INFO
    assert check_verbose(False) == logging.WARNING
    assert check_verbose(None) == logging.WARNING

    # invalids
    with pytest.raises(TypeError, match="must be an instance of"):
        check_verbose(("INFO",))
    with pytest.raises(ValueError, match="Invalid value"):
        check_verbose("101")
    with pytest.raises(ValueError, match="negative integer, -101 is invalid."):
        check_verbose(-101)


def test_ensure_path():
    """Test ensure_path checker."""
    # valids
    cwd = Path.cwd()
    path = ensure_path(cwd, must_exist=False)
    assert isinstance(path, Path)
    path = ensure_path(cwd, must_exist=True)
    assert isinstance(path, Path)
    path = ensure_path(str(cwd), must_exist=False)
    assert isinstance(path, Path)
    path = ensure_path(str(cwd), must_exist=True)
    assert isinstance(path, Path)
    path = ensure_path("101", must_exist=False)
    assert isinstance(path, Path)

    with pytest.raises(FileNotFoundError, match="does not exist."):
        ensure_path("101", must_exist=True)

    # invalids
    with pytest.raises(TypeError, match="'101' is invalid"):
        ensure_path(101, must_exist=False)

    class Foo:
        def __str__(self):
            pass

    with pytest.raises(TypeError, match="path is invalid"):
        ensure_path(Foo(), must_exist=False)
