from __future__ import annotations

from click.testing import CliRunner

from mne_lsl._commands.main import run


def test_main() -> None:
    """Test the main package entry-point."""
    runner = CliRunner()
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0
    assert "Options:" in result.output
    assert "Commands:" in result.output
