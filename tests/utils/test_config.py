from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

import mne_lsl
from mne_lsl.utils.config import _get_gpu_info, sys_info


def test_sys_info() -> None:
    """Test info-showing utility."""
    out = StringIO()
    sys_info(fid=out)
    value = out.getvalue()
    out.close()
    assert "Platform:" in value
    assert "Executable:" in value
    assert "CPU:" in value
    assert "Physical cores:" in value
    assert "Logical cores" in value
    assert "RAM:" in value
    assert "SWAP:" in value

    assert "numpy" in value
    assert "psutil" in value

    assert "style" not in value
    assert "test" not in value


@pytest.mark.skipif(
    not (Path(mne_lsl.__file__).parents[2] / "pyproject.toml").exists(),
    reason="not editable install",
)
def test_sys_info_developer() -> None:
    """Test info-showing utility, with developer dependencies."""
    out = StringIO()
    sys_info(fid=out, developer=True)
    value = out.getvalue()
    out.close()
    assert "test" in value


def test_gpu_info() -> None:
    """Test getting GPU info."""
    pytest.importorskip("pyvista")
    version, renderer = _get_gpu_info()
    assert version is not None
    assert renderer is not None
