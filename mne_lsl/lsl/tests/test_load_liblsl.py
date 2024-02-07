import platform
from ctypes import c_void_p, sizeof
from itertools import chain
from pathlib import Path
from shutil import copy

import pooch
import pytest
import requests

from mne_lsl import __version__
from mne_lsl.lsl.load_liblsl import (
    _PLATFORM,
    _PLATFORM_SUFFIXES,
    _SUPPORTED_DISTRO,
    _attempt_load_liblsl,
    _fetch_liblsl,
    _is_valid_libpath,
    _is_valid_version,
    _load_liblsl_mne_lsl,
    _pooch_processor_liblsl,
)


@pytest.fixture(scope="module")
def _download_liblsl_outdated(tmp_path_factory) -> Path:
    """Fixture to download an outdated liblsl version."""
    assert _PLATFORM in _PLATFORM_SUFFIXES  # test OS-detection
    if _PLATFORM == "darwin" and platform.processor() == "i386":
        asset = dict(
            name="liblsl-1.14.0-OSX_amd64.tar.bz2",
            browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.14.0/liblsl-1.14.0-OSX_amd64.tar.bz2",  # noqa: E501
            known_hash="c1f9004243db49885b18884b39d793f66c94e45d52a7bde12c51893e74db337d",
        )
    elif _PLATFORM == "windows" and sizeof(c_void_p) == 8:
        asset = dict(
            name="liblsl-1.14.0-Win_amd64.zip",
            browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.14.0/liblsl-1.14.0-Win_amd64.zip",  # noqa: E501
            known_hash="75ec445e9e9b23b15400322fffa06666098fce42e706afa763e77abdbea87e52",
        )
    else:
        pytest.skip(reason="Unsupported platform for this test.")

    try:
        libpath = pooch.retrieve(
            url=asset["browser_download_url"],
            fname=asset["name"],
            path=tmp_path_factory.mktemp("data"),
            processor=_pooch_processor_liblsl,
            known_hash=asset["known_hash"],
        )
    except ValueError:
        pytest.skip(reason="Unable to download the outdated liblsl.")
    return Path(libpath)


@pytest.fixture(scope="function")
def liblsl_outdated(tmp_path, _download_liblsl_outdated) -> Path:
    """Fixture to provide an outdated liblsl version."""
    copy(_download_liblsl_outdated, tmp_path / _download_liblsl_outdated.name)
    return tmp_path / _download_liblsl_outdated.name


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
    _PLATFORM == "windows",
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
    _PLATFORM == "windows",
    reason="PermissionError: [WinError 5] Access is denied (on Path.unlink(...)).",
)
@pytest.mark.skipif(
    _PLATFORM == "darwin" and platform.processor() == "arm",
    reason="Automatic bypass with version 1.16.0 for M1/M2 macOS",
)
@pytest.mark.xfail(raises=KeyError, reason="403 Forbidden Error on GitHub API request.")
def test_fetch_liblsl_outdated(tmp_path):
    """Test fetching an outdated version of liblsl."""
    with pytest.raises(RuntimeError, match="is outdated. The version is"):
        _fetch_liblsl(
            folder=tmp_path,
            url="https://api.github.com/repos/sccn/liblsl/releases/tags/v1.14.0",
        )


def test_liblsl_outdated(liblsl_outdated):
    """Test loading an outdated version of liblsl."""
    libpath, version = _attempt_load_liblsl(liblsl_outdated)
    assert isinstance(libpath, str)
    assert isinstance(version, int)
    assert _is_valid_libpath(libpath)
    assert not _is_valid_version(libpath, version, issue_warning=False)
    with pytest.warns(RuntimeWarning, match="is outdated. The version is"):
        _is_valid_version(libpath, version, issue_warning=True)


@pytest.mark.skipif(
    _PLATFORM == "windows",
    reason="PermissionError: [WinError 5] Access is denied (on Path.unlink(...)).",
)
def test_liblsl_outdated_mne_folder(liblsl_outdated):
    """Test loading an outdated version of liblsl in the MNE folder."""
    assert liblsl_outdated.exists()
    with pytest.warns(RuntimeWarning, match="is outdated. The version is"):
        _load_liblsl_mne_lsl(folder=liblsl_outdated.parent)
    assert not liblsl_outdated.exists()


def test_is_valid_libpath(tmp_path):
    """Test _is_valid_libpath."""
    fname = str(tmp_path / "101.txt")
    with open(fname, "w") as file:
        file.write("101")
    with pytest.warns(RuntimeWarning, match="different from the expected extension"):
        valid = _is_valid_libpath(fname)
    assert not valid

    fname = str(tmp_path / f"101.{_PLATFORM_SUFFIXES[_PLATFORM]}")
    with pytest.warns(RuntimeWarning, match="does not exist"):
        valid = _is_valid_libpath(fname)
    assert not valid
