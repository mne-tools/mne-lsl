from platform import system

import pytest

from bsl.triggers import ParallelPortTrigger


def test_infer_port_type():
    """Test port type inference patterns."""
    assert ParallelPortTrigger._infer_port_type("arduino") == "arduino"

    if system() == "Linux":
        assert ParallelPortTrigger._infer_port_type("/dev/parport0") == "pport"
        assert ParallelPortTrigger._infer_port_type("/dev/parport1") == "pport"
        assert ParallelPortTrigger._infer_port_type("/dev/ttyACM0") == "arduino"
        assert ParallelPortTrigger._infer_port_type("/dev/ttyACM1") == "arduino"
        with pytest.raises(RuntimeError, match="Could not infer the port type"):
            ParallelPortTrigger._infer_port_type("101")

    elif system() == "Windows":
        assert ParallelPortTrigger._infer_port_type("COM7") == "arduino"
        assert ParallelPortTrigger._infer_port_type("COM8") == "arduino"
        assert ParallelPortTrigger._infer_port_type(0x4fb8) == "pport"


def test_search_arduino():
    """Test arduino detection."""
    with pytest.raises(IOError, match="No arduino card was found."):
        ParallelPortTrigger._search_arduino()
