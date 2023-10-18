from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
from typing import TYPE_CHECKING

import numpy as np
from mne import create_info
from mne.io import RawArray
from pytest import fixture

from mne_lsl.datasets import testing
from mne_lsl.player import PlayerLSL

if TYPE_CHECKING:
    from pathlib import Path


def pytest_configure(config):
    """Configure pytest options."""
    if os.getenv("MNE_IGNORE_WARNINGS_IN_TESTS", "") != "true":
        first_kind = "error"
    else:
        first_kind = "always"
    warning_lines = f"    {first_kind}::"
    warning_lines += r"""
    # numpy 2.0 <-> SciPy
    ignore:numpy\.core\._multiarray_umath.*:DeprecationWarning
    ignore:numpy\.core\.multiarray is deprecated.*:DeprecationWarning
    ignore:datetime\.datetime\.utcfromtimestamp.*:DeprecationWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)

@fixture(scope="module")
def mock_lsl_stream():
    """Create a mock LSL stream for testing."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    with PlayerLSL(fname, "Player-pytest", chunk_size=16):
        yield


@fixture(scope="session")
def _integer_raw(tmp_path_factory) -> Path:
    """Create a Raw object with each channel containing its idx continuously."""
    info = create_info(5, 1000, "eeg")
    data = np.full((5, 1000), np.arange(5).reshape(-1, 1))
    raw = RawArray(data, info)
    fname = tmp_path_factory.mktemp("data") / "int-raw.fif"
    raw.save(fname)
    return fname


@fixture(scope="module")
def mock_lsl_stream_int(_integer_raw):
    """Create a mock LSL stream streaming the channel number continuously."""
    with PlayerLSL(_integer_raw, "Player-integers-pytest", chunk_size=16):
        yield
