import platform

from mne_lsl.lsl.load_liblsl import _PLATFORM_SUFFIXES


def test_os_detection():
    """Test OS detection for MNE-LSL's supported OS.

    Make sure platform.system() returns a valid entry.
    """
    assert platform.system() in _PLATFORM_SUFFIXES
