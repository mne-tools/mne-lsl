from __future__ import annotations

import pytest
from click.testing import CliRunner

from mne_lsl._commands.sys_info import run


@pytest.mark.parametrize("developer", [False, True])
def test_sys_info(developer: bool) -> None:
    """Test the system information entry-point."""
    runner = CliRunner()
    result = runner.invoke(run, ["--developer"] if developer else [])
    assert result.exit_code == 0
    assert "Platform:" in result.output
    assert "Python:" in result.output
    assert "Executable:" in result.output
    assert "Core dependencies" in result.output
    if developer:
        assert "Developer 'style' dependencies" in result.output
        assert "Developer 'test' dependencies" in result.output
