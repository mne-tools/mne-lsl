import platform

from bsl.lsl.load_liblsl import _PLATFORM_SUFFIXES


def test_os_detection():
    """Test OS detection for BSL's supported OS.

    Make sure platform.system() returns a valid entry.
    """
    assert platform.system() in _PLATFORM_SUFFIXES
