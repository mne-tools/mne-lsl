from __future__ import annotations

import inspect
import logging
import os
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import Annotations
from mne import set_log_level as set_log_level_mne
from mne.io import read_raw_fif

from mne_lsl.datasets import testing
from mne_lsl.lsl import StreamInlet, StreamOutlet
from mne_lsl.utils._checks import check_verbose
from mne_lsl.utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path

    from mne.io import BaseRaw

# Set debug logging in LSL, e.g.:
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         udp_server.cpp:88       2| P_test_stream_add_reference_channels[1s]: Started multicast udp server at ff05:113d:6fdd:2c17:a643:ffe2:1bd1:3cd2 port 16571 (addr 0x43f93a0)  # noqa: E501
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         tcp_server.cpp:160      1| Created IPv4 TCP acceptor for P_test_stream_add_reference_channels[1s] @ port 16572  # noqa: E501
# 2023-10-20 09:38:21.639 (   9.656s) [pytest          ]         tcp_server.cpp:171      1| Created IPv6 TCP acceptor for P_test_stream_add_reference_channels[1s] @ port 16578  # noqa: E501
# 2023-10-20 09:38:21.648 (   9.665s) [IO_P_test_stre  ]         udp_server.cpp:136      3| 0x43e9310 query matches, replying to port 16574  # noqa: E501
lsl_cfg = NamedTemporaryFile("w", prefix="lsl", suffix=".cfg", delete=False)
if "LSLAPICFG" not in os.environ:
    verbose = os.getenv("MNE_LSL_LOG_LEVEL", 2)
    try:
        verbose = int(verbose)
    except ValueError:
        pass
    verbose = check_verbose(verbose)
    # LSL logs use '-2- for errors, -1 for warnings, 0  for information and then
    # 1-9 for increasingly less important details.
    if logging.ERROR <= verbose:
        level = -2
    elif logging.WARNING <= verbose:
        level = -1
    elif logging.INFO <= verbose:
        level = 0
    else:
        level = 2
    with lsl_cfg as fid:
        fid.write(f"[log]\nlevel = {level}\n\n[multicast]\nResolveScope = link")
    os.environ["LSLAPICFG"] = lsl_cfg.name


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest options."""
    for marker in ("slow",):
        config.addinivalue_line("markers", marker)
    if "MNE_LSL_RAISE_STREAM_ERRORS" not in os.environ:
        os.environ["MNE_LSL_RAISE_STREAM_ERRORS"] = "true"
    first_kind = (
        "error" if os.getenv("MNE_IGNORE_WARNINGS_IN_TESTS", "") != "true" else "always"
    )
    warning_lines = f"    {first_kind}::"
    warning_lines += r"""
    # Pooch tar
    ignore:Python 3.14 will, by default.*:DeprecationWarning
    # Matplotlib deprecation issued in VSCode test debugger
    ignore:.*interactive_bk.*:matplotlib._api.deprecation.MatplotlibDeprecationWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
    set_log_level_mne("WARNING")  # MNE logger
    logger.propagate = True


def pytest_sessionfinish(session, exitstatus) -> None:
    """Clean up the pytest session."""
    try:
        os.unlink(lsl_cfg.name)
    except Exception:
        pass


def _closer():
    """Delete inlets and outlets if present.

    We cannot rely on just "del inlet" / "del outlet" because Python's garbage collector
    can run whenever it feels like it, and AFAIK the garbage collection order is not
    guaranteed. So let's explicitly __del__ ourselves, knowing that our __del__s are
    smart enough to be no-ops if called more than once.
    """
    loc = inspect.currentframe().f_back.f_locals
    inlets, outlets = [], []
    for var in loc.values():  # go through the frame
        if isinstance(var, StreamInlet):
            inlets.append(var)
        elif isinstance(var, StreamOutlet):
            outlets.append(var)
    # delete inlets before outlets
    for inlet in inlets:
        inlet._del()
        del inlet
    inlets.clear()
    for outlet in outlets:
        outlet._del()
        del outlet
    outlets.clear()


@pytest.fixture()
def close_io():
    """Return function that will close inlets and outlets if present."""
    return _closer


@pytest.fixture(scope="session")
def fname(tmp_path_factory) -> Path:
    """Yield fname of a file with sample numbers in the first channel."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    raw = read_raw_fif(fname, preload=True)  # 67 channels x 2049 samples -> 2 seconds
    raw._data[0] = np.arange(len(raw.times))
    raw.rename_channels({raw.ch_names[0]: "Samples"})
    raw.set_channel_types({raw.ch_names[0]: "misc"}, on_unit_change="ignore")
    fname_mod = tmp_path_factory.mktemp("data") / "sample-eeg-ant-raw.fif"
    raw.save(fname_mod)
    return fname_mod


@pytest.fixture()
def raw(fname: Path) -> BaseRaw:
    """Return the raw file corresponding to fname."""
    return read_raw_fif(fname, preload=True)


@pytest.fixture()
def raw_annotations(raw: BaseRaw) -> BaseRaw:
    """Return a raw file with annotations."""
    annotations = Annotations(
        onset=[0.1, 0.4, 0.5, 0.8, 0.95, 1.1, 1.3],
        duration=[0.2, 0.2, 0.2, 0.1, 0.05, 0.4, 0.55],
        description=["test1", "test1", "test1", "test2", "test3", "bad_test", "test1"],
    )
    raw.set_annotations(annotations)
    return raw


@pytest.fixture(scope="session")
def chunk_size() -> int:
    """Return the chunk size for testing."""
    return 200
