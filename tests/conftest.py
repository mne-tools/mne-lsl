from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import Annotations, get_config
from mne import set_log_level as set_log_level_mne
from mne.io import read_raw_fif

from mne_lsl.datasets import testing
from mne_lsl.lsl import StreamInlet, StreamOutlet, set_config_content
from mne_lsl.utils._checks import check_verbose
from mne_lsl.utils.logs import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mne.io import BaseRaw

if "LSLAPICFG" not in os.environ:
    verbose = get_config("MNE_LSL_LOG_LEVEL", default=2)
    try:
        verbose = int(verbose)
    except ValueError:
        pass
    verbose = check_verbose(verbose)
    # LSL logs use '-2' for errors, -1 for warnings, 0 for information and then
    # 1-9 for increasingly less important details.
    if logging.ERROR <= verbose:
        level = -2
    elif logging.WARNING <= verbose:
        level = -1
    elif logging.INFO <= verbose:
        level = 0
    else:
        level = 2
    # configure liblsl directly instead of through a temporary file and the 'LSLAPICFG'
    # environment variable (requires liblsl >= 1.17.7).
    set_config_content(f"[log]\nlevel = {level}\n\n[multicast]\nResolveScope = link")


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
    # Pillow deprecation issued from matplotlib
    ignore:'mode' parameter is deprecated.*:DeprecationWarning
    # tkinter
    ignore:Exception ignored in.*__del__.*:pytest.PytestUnraisableExceptionWarning
    # NumPy deprecation hitting MNE-Python: github.com/mne-tools/mne-python/pull/13585
    ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
    set_log_level_mne("WARNING")  # MNE logger
    logger.propagate = True


def _closer() -> None:
    """Delete inlets and outlets if present.

    We cannot rely on just "del inlet" / "del outlet" because Python's garbage collector
    can run whenever it feels like it, and AFAIK the garbage collection order is not
    guaranteed. So let's explicitly __del__ ourselves, knowing that our __del__s are
    smart enough to be no-ops if called more than once.
    """
    loc = inspect.currentframe().f_back.f_locals
    inlets: list[StreamInlet] = []
    outlets: list[StreamOutlet] = []
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


@pytest.fixture
def close_io() -> Callable[[], None]:
    """Return function that will close inlets and outlets if present."""
    return _closer


@pytest.fixture(scope="session")
def fname(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Yield fname of a file with sample numbers in the first channel."""
    fname = testing.data_path() / "sample-eeg-ant-raw.fif"
    raw = read_raw_fif(fname, preload=True)  # 67 channels x 2049 samples -> 2 seconds
    raw._data[0] = np.arange(len(raw.times))
    raw.rename_channels({raw.ch_names[0]: "Samples"})
    raw.set_channel_types({raw.ch_names[0]: "misc"}, on_unit_change="ignore")
    fname_mod = tmp_path_factory.mktemp("data") / "sample-eeg-ant-raw.fif"
    raw.save(fname_mod)
    return fname_mod


@pytest.fixture
def raw(fname: Path) -> BaseRaw:
    """Return the raw file corresponding to fname."""
    return read_raw_fif(fname, preload=True)


@pytest.fixture
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
