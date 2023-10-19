from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
from typing import TYPE_CHECKING

import numpy as np
from mne import create_info, set_log_level
from mne.io import Raw, RawArray, read_raw_fif
from pytest import fixture

from mne_lsl.datasets import testing
from mne_lsl.player import PlayerLSL

if TYPE_CHECKING:
    from pathlib import Path


def pytest_configure(config):
    """Configure pytest options."""
    for marker in ("slow",):
        config.addinivalue_line("markers", marker)
    if "MNE_LSL_RAISE_STREAM_ERRORS" not in os.environ:
        os.environ["MNE_LSL_RAISE_STREAM_ERRORS"] = "true"
    if os.getenv("MNE_IGNORE_WARNINGS_IN_TESTS", "") != "true":
        first_kind = "error"
    else:
        first_kind = "always"
    warning_lines = f"    {first_kind}::"
    warning_lines += r"""
    # numpy 2.0 <-> SciPy
    ignore:numpy\.core\._multiarray_umath.*:DeprecationWarning
    ignore:numpy\.core\.multiarray is deprecated.*:DeprecationWarning
    ignore:numpy\.core\.numeric is deprecated.*:DeprecationWarning
    ignore:datetime\.datetime\.utcfromtimestamp.*:DeprecationWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
    set_log_level("WARNING")  # MNE logger


@fixture(scope="session")
def fname(tmp_path_factory) -> Path:
    """Yield fname of a file with sample numbers in the last channel."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    raw = read_raw_fif(fname, preload=True)
    raw._data[-1] = np.arange(1, len(raw.times) + 1) * 1e-6
    fname_mod = tmp_path_factory.mktemp("data") / "sample-eeg-ant-raw.fif"
    raw.save(fname_mod)
    return fname_mod


@fixture(scope="function")
def raw(fname) -> Raw:
    """Return the raw file corresponding to fname."""
    return read_raw_fif(fname, preload=True)


@fixture(scope="function")
def mock_lsl_stream(fname, request):
    """Create a mock LSL stream for testing."""
    name = f"P_{request.node.name}"
    with PlayerLSL(fname, name, chunk_size=16) as player:
        yield player


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
def mock_lsl_stream_int(_integer_raw, request):
    """Create a mock LSL stream streaming the channel number continuously."""
    name = f"P_{request.node.name}"
    with PlayerLSL(_integer_raw, name, chunk_size=16) as player:
        yield player
