from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import mne_lsl
from mne_lsl._commands.sys_info import run


def test_sys_info() -> None:
    """Test the system information entry-point."""
    runner = CliRunner()
    result = runner.invoke(run, [])
    assert result.exit_code == 0
    assert "Platform:" in result.output
    assert "Python:" in result.output
    assert "Executable:" in result.output
    assert "Core dependencies" in result.output


@pytest.mark.skipif(
    not (Path(mne_lsl.__file__).parents[2] / "pyproject.toml").exists(),
    reason="not editable install",
)
def test_sys_info_developer() -> None:
    """Test the system information entry-point with developer infos."""
    runner = CliRunner()
    result = runner.invoke(run, ["--developer"])
    assert result.exit_code == 0
    assert "Developer 'test' dependencies" in result.output
