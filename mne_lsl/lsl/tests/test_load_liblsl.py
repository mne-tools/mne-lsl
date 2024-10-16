import platform
from ctypes import c_void_p, sizeof
from pathlib import Path
from shutil import copy

import pooch
import pytest

from mne_lsl.lsl.load_liblsl import (
    _PLATFORM,
    _PLATFORM_SUFFIXES,
    _attempt_load_liblsl,
    _fetch_liblsl,
    _is_valid_libpath,
    _is_valid_version,
    _load_liblsl_mne_lsl,
    _pooch_processor_liblsl,
)


@pytest.fixture(scope="module")
def download_liblsl_outdated(tmp_path_factory) -> Path:
    """Fixture to download an outdated liblsl version."""
    assert _PLATFORM in _PLATFORM_SUFFIXES  # test OS-detection
    if _PLATFORM == "darwin" and platform.processor() == "i386":
        asset = dict(
            name="liblsl-1.14.0-OSX_amd64.tar.bz2",
            browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.14.0/liblsl-1.14.0-OSX_amd64.tar.bz2",  # noqa: E501
            known_hash="c1f9004243db49885b18884b39d793f66c94e45d52a7bde12c51893e74db337d",  # noqa: E501
        )
    elif _PLATFORM == "windows" and sizeof(c_void_p) == 8:
        asset = dict(
            name="liblsl-1.14.0-Win_amd64.zip",
            browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.14.0/liblsl-1.14.0-Win_amd64.zip",  # noqa: E501
            known_hash="75ec445e9e9b23b15400322fffa06666098fce42e706afa763e77abdbea87e52",  # noqa: E501
        )
    elif _PLATFORM == "linux":
        import distro

        if distro.codename() == "bionic":
            asset = dict(
                name="liblsl-1.14.0-bionic_amd64.deb",
                browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.14.0/liblsl-1.14.0-bionic_amd64.deb",  # noqa: E501
                known_hash="b2b882414a73acba4e7ab14361a6d541cc1a3774536e08471b346a89b4b557bb",  # noqa: E501
            )
        elif distro.codename() == "focal":
            asset = dict(
                name="liblsl-1.14.0-focal_amd64.deb",
                browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.14.0/liblsl-1.14.0-focal_amd64.deb",  # noqa: E501
                known_hash="b3a0c0e40746f126d77212400e1e23b2f3225bf0458215731b08b1bde631f15b",  # noqa: E501
            )
        else:
            pytest.skip(reason=f"Unsupported linux distribution '{distro.codename()}'.")
    else:
        pytest.skip(reason=f"Unsupported platform '{_PLATFORM}'.")

    try:
        libpath = pooch.retrieve(
            url=asset["browser_download_url"],
            fname=asset["name"],
            path=tmp_path_factory.mktemp("liblsl_outdated"),
            processor=_pooch_processor_liblsl,
            known_hash=asset["known_hash"],
        )
    except ValueError:
        pytest.skip(reason="Unable to download the outdated liblsl (invalid hash).")
    return Path(libpath)


@pytest.fixture()
def liblsl_outdated(tmp_path, download_liblsl_outdated) -> Path:
    """Fixture to provide an outdated liblsl version."""
    copy(download_liblsl_outdated, tmp_path / download_liblsl_outdated.name)
    return tmp_path / download_liblsl_outdated.name


@pytest.mark.skipif(
    _PLATFORM == "linux",
    reason="Runner ubuntu-latest runs on 24.04 and LSL did not release yet for it.",
)
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
    with open(fname.parent / f"test{_PLATFORM_SUFFIXES[_PLATFORM]}", "w") as file:
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
    if _PLATFORM == "linux":
        import distro

        if distro.codename() not in ("bionic", "focal"):
            pytest.skip("Unsupported Ubuntu version.")
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
