from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from mne import create_info
from mne import set_log_level as set_log_level_mne
from mne.io import Raw, RawArray, read_raw_fif
from pytest import fixture

from mne_lsl import set_log_level
from mne_lsl.datasets import testing  # ignore: E402

# Set debug logging in LSL, e.g.:
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         udp_server.cpp:88       2| P_test_stream_add_reference_channels[1s]: Started multicast udp server at ff05:113d:6fdd:2c17:a643:ffe2:1bd1:3cd2 port 16571 (addr 0x43f93a0)  # noqa: E501
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         tcp_server.cpp:160      1| Created IPv4 TCP acceptor for P_test_stream_add_reference_channels[1s] @ port 16572  # noqa: E501
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         tcp_server.cpp:171      1| Created IPv6 TCP acceptor for P_test_stream_add_reference_channels[1s] @ port 16578  # noqa: E501
# 2023-10-20 09:38:21.648 (   9.665s) [IO_P_test_stre  ]         udp_server.cpp:136      3| 0x43e9310 query matches, replying to port 16574  # noqa: E501
lsl_cfg = tempfile.NamedTemporaryFile("w", prefix="lsl", suffix=".cfg", delete=False)
if "LSLAPICFG" not in os.environ:
    level = int(os.getenv("MNE_LSL_LOG_LEVEL", "2"))
    with lsl_cfg as fid:
        fid.write(
            f"""
[log]
level = {level}
"""
        )
    os.environ["LSLAPICFG"] = lsl_cfg.name

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
    set_log_level_mne("WARNING")  # MNE logger
    set_log_level("DEBUG")  # MNE-lsl logger


def pytest_sessionfinish(session, exitstatus):
    try:
        os.unlink(lsl_cfg.name)
    except Exception:
        pass


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
    # Nest PlayerLSL import so temp config gets written first
    from mne_lsl.player import PlayerLSL  # ignore: E402

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
    from mne_lsl.player import PlayerLSL  # ignore: E402

    name = f"P_{request.node.name}"
    with PlayerLSL(_integer_raw, name, chunk_size=16) as player:
        yield player
