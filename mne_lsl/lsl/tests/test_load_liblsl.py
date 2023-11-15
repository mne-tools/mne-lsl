import pytest
import requests

from mne_lsl.lsl.load_liblsl import _PLATFORM, _PLATFORM_SUFFIXES, _SUPPORTED_DISTRO


def test_os_detection():
    """Test OS detection for MNE-LSL's supported OS.

    Make sure platform.system() returns a valid entry.
    """
    assert _PLATFORM in _PLATFORM_SUFFIXES


@pytest.mark.xfail(raises=KeyError, reason="403 Forbidden Error on GitHub API request.")
def test_distro_support():
    """Test that the variables are in sync with the latest liblsl release."""
    response = requests.get("https://api.github.com/repos/sccn/liblsl/releases/latest")
    assets = [
        elt["name"]
        for elt in response.json()["assets"]
        if "liblsl" in elt["name"] and not any(x in elt["name"] for x in ("OSX", "Win"))
    ]
    assets = sorted(assets)
    assert len(assets) == 3, f"Supported liblsl are {', '.join(assets)}."
    assert "bionic" in assets[0]  # 18.04 LTS
    assert "focal" in assets[1]  # 20.04 LTS
    assert "jammy" in assets[2]  # 22.04 LTS
    # confirm that it matches _SUPPORTED_DISTRO
    assert _SUPPORTED_DISTRO["ubuntu"] == ("18.04", "20.04", "22.04")
