from os import makedirs

import pytest

from mne_lsl.utils._path import walk


def test_walk(tmp_path):
    """Test walk generator."""
    fname1 = tmp_path / "file1"
    with open(fname1, "w"):
        pass
    fname2 = tmp_path / "file2"
    with open(fname2, "w"):
        pass
    makedirs(tmp_path / "dir1" / "dir2")
    fname3 = tmp_path / "dir1" / "file3"
    with open(fname3, "w"):
        pass
    fname4 = tmp_path / "dir1" / "dir2" / "file1"
    with open(fname4, "w"):
        pass
    files = list(walk(tmp_path))
    assert all(fname in files for fname in (fname1, fname2, fname3, fname4))
    with pytest.raises(RuntimeError, match="not a directory"):
        list(walk(fname1))
