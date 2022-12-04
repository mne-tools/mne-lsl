import platform


def test_os_detection():
    """Test OS detection for BSL's supported OS.

    Make sure platform.system() returns a valid entry.
    """
    assert platform.system() in ("Windows", "Darwin", "Linux")
