from __future__ import annotations  # c.f. PEP 563, PEP 649

import inspect
import os
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np
from mne import set_log_level as set_log_level_mne
from mne.io import Raw, read_raw_fif
from pytest import fixture

from mne_lsl import set_log_level
from mne_lsl.datasets import testing

# Set debug logging in LSL, e.g.:
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         udp_server.cpp:88       2| P_test_stream_add_reference_channels[1s]: Started multicast udp server at ff05:113d:6fdd:2c17:a643:ffe2:1bd1:3cd2 port 16571 (addr 0x43f93a0)  # noqa: E501
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         tcp_server.cpp:160      1| Created IPv4 TCP acceptor for P_test_stream_add_reference_channels[1s] @ port 16572  # noqa: E501
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         tcp_server.cpp:171      1| Created IPv6 TCP acceptor for P_test_stream_add_reference_channels[1s] @ port 16578  # noqa: E501
# 2023-10-20 09:38:21.648 (   9.665s) [IO_P_test_stre  ]         udp_server.cpp:136      3| 0x43e9310 query matches, replying to port 16574  # noqa: E501
lsl_cfg = NamedTemporaryFile("w", prefix="lsl", suffix=".cfg", delete=False)
if "LSLAPICFG" not in os.environ:
    level = int(os.getenv("MNE_LSL_LOG_LEVEL", "2"))
    with lsl_cfg as fid:
        fid.write(f"[log]\nlevel = {level}\n\n[multicast]\nResolveScope = link")
    os.environ["LSLAPICFG"] = lsl_cfg.name

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import Config


def pytest_configure(config: Config) -> None:
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
    # Pooch tar
    ignore:Python 3.14 will, by default.*:DeprecationWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
    set_log_level_mne("WARNING")  # MNE logger
    set_log_level("DEBUG")  # MNE-lsl logger


def pytest_sessionfinish(session, exitstatus) -> None:
    """Clean up the pytest session."""
    try:
        os.unlink(lsl_cfg.name)
    except Exception:
        pass


def _closer():
    """Delete inlet, outlet, and player vars if present.

    We cannot rely on just "del inlet" / "del outlet" because Python's garbage collector
    can run whenever it feels like it, and AFAIK the garbage collection order is not
    guaranteed. So let's explicitly __del__ ourselves, knowing that our __del__s are
    smart enough to be no-ops if called more than once.
    """
    loc = inspect.currentframe().f_back.f_locals
    for name in ("inlet", "outlet"):
        if name in loc:
            loc[name].__del__()
    if "player" in loc:
        loc["player"].stop()


@fixture(scope="function")
def close_io():
    """Return function that will close inlet, outlet, and player vars if present."""
    return _closer


@fixture(scope="session")
def fname(tmp_path_factory) -> Path:
    """Yield fname of a file with sample numbers in the first channel."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    raw = read_raw_fif(fname, preload=True)
    raw._data[0] = np.arange(len(raw.times))
    raw.rename_channels({raw.ch_names[0]: "Samples"})
    raw.set_channel_types({raw.ch_names[0]: "misc"}, on_unit_change="ignore")
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
    # nest the PlayerLSL import to first write the temporary LSL configuration file
    from mne_lsl.player import PlayerLSL  # noqa: E402

    name = f"P_{request.node.name}"
    with PlayerLSL(fname, name) as player:
        yield player
