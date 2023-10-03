from platform import system

import pytest

from mne_lsl.triggers import ParallelPortTrigger


@pytest.mark.skipif(system() != "Linux", reason="requires Linux")
def test_infer_port_type_linux():
    """Test port type inference patterns on linux."""
    assert ParallelPortTrigger._infer_port_type("arduino") == "arduino"
    assert ParallelPortTrigger._infer_port_type("/dev/parport0") == "pport"
    assert ParallelPortTrigger._infer_port_type("/dev/parport1") == "pport"
    assert ParallelPortTrigger._infer_port_type("/dev/ttyACM0") == "arduino"
    assert ParallelPortTrigger._infer_port_type("/dev/ttyACM1") == "arduino"
    with pytest.raises(RuntimeError, match="provide the 'port_type' argument"):
        ParallelPortTrigger._infer_port_type("101")
    with pytest.raises(TypeError, match="provided as a string"):
        ParallelPortTrigger._infer_port_type(0x4FB8)


@pytest.mark.skipif(system() != "Windows", reason="requires Windows")
def test_infer_port_type_windows():
    """Test port type inference patterns on Windows."""
    assert ParallelPortTrigger._infer_port_type("arduino") == "arduino"
    assert ParallelPortTrigger._infer_port_type("COM7") == "arduino"
    assert ParallelPortTrigger._infer_port_type("COM8") == "arduino"
    assert ParallelPortTrigger._infer_port_type(0x4FB8) == "pport"


@pytest.mark.skipif(system() != "Darwin", reason="requires macOS")
def test_infer_port_type_macos():
    """Test port type inference patterns on Windows."""
    assert ParallelPortTrigger._infer_port_type("arduino") == "arduino"
    with pytest.raises(RuntimeError, match="macOS does not support"):
        ParallelPortTrigger._infer_port_type("/dev/parport0")
    with pytest.raises(RuntimeError, match="macOS does not support"):
        ParallelPortTrigger._infer_port_type(0x4FB8)


def test_search_arduino():
    """Test arduino detection."""
    with pytest.raises(IOError, match="No arduino card was found."):
        ParallelPortTrigger._search_arduino()
