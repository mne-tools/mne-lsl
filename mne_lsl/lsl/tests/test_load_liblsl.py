import platform
from itertools import chain
from pathlib import Path

import pytest
import requests

from mne_lsl import __version__
from mne_lsl.lsl.load_liblsl import (
    _PLATFORM,
    _PLATFORM_SUFFIXES,
    _SUPPORTED_DISTRO,
    _fetch_liblsl,
    _load_liblsl_mne_lsl,
)


def test_os_detection():
    """Test OS detection for MNE-LSL's supported OS.

    Make sure platform.system() returns a valid entry.
    """
    assert _PLATFORM in _PLATFORM_SUFFIXES


@pytest.mark.xfail(raises=KeyError, reason="403 Forbidden Error on GitHub API request.")
def test_distro_support():
    """Test that the variables are in sync with the latest liblsl release."""
    response = requests.get(
        "https://api.github.com/repos/sccn/liblsl/releases/latest",
        timeout=15,
        headers={"user-agent": f"mne-lsl/{__version__}"},
    )
    assets = [
        elt["name"]
        for elt in response.json()["assets"]
        if "liblsl" in elt["name"] and not any(x in elt["name"] for x in ("OSX", "Win"))
    ]
    assets = sorted(assets)
    assert len(assets) == len(
        list(chain(*_SUPPORTED_DISTRO.values()))
    ), f"Supported liblsl are {', '.join(assets)}."
    for key in ("bookworm", "bionic", "focal", "jammy"):
        assert any(key in asset for asset in assets)
    # confirm that it matches _SUPPORTED_DISTRO
    assert _SUPPORTED_DISTRO["debian"] == ("12",)
    assert _SUPPORTED_DISTRO["ubuntu"] == ("18.04", "20.04", "22.04")


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="PermissionError: [WinError 5] Access is denied (on Path.unlink(...)).",
)
@pytest.mark.xfail(raises=KeyError, reason="403 Forbidden Error on GitHub API request.")
def test_fetch_liblsl(tmp_path):
    """Test on-the-fly fetch of liblsl."""
    libpath = _fetch_liblsl(folder=tmp_path)
    assert Path(libpath).exists()
    # don't re-download if it's already present
    libpath2 = _load_liblsl_mne_lsl(folder=tmp_path)
    assert libpath == libpath2
    assert Path(libpath).exists()
    # delete the file and try again
    fname = Path(libpath)
    fname.unlink(missing_ok=False)
    libpath2 = _fetch_liblsl(folder=tmp_path)
    assert libpath == libpath2
    assert Path(libpath).exists()
    # replace the file and try to reload it
    fname.unlink(missing_ok=False)
    with open(fname, "w") as file:
        file.write("101")
    with pytest.warns(RuntimeWarning, match="could not be loaded. It will be removed."):
        libpath2 = _load_liblsl_mne_lsl(folder=tmp_path)
    assert libpath2 is None
    # the last call to _load_liblsl_mne_lsl should have removed the file
    libpath2 = _fetch_liblsl(folder=tmp_path)
    assert libpath == libpath2
    assert Path(libpath).exists()
    # test invalid fetch folder
    with pytest.raises(RuntimeError, match="is a file. Please provide a directory"):
        _fetch_liblsl(folder=fname)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="PermissionError: [WinError 5] Access is denied (on Path.unlink(...)).",
)
@pytest.mark.xfail(raises=KeyError, reason="403 Forbidden Error on GitHub API request.")
def test_fetch_liblsl_outdated(tmp_path):
    """Test fetching an outdated version of liblsl."""
    with pytest.raises(RuntimeError, match="is outdated. The version is"):
        _fetch_liblsl(
            folder=tmp_path,
            url="https://api.github.com/repos/sccn/liblsl/releases/tags/v1.14.0",
        )
