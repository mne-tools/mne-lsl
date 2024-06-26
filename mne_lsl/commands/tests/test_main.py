from click.testing import CliRunner

from ..main import run


def test_main():
    """Test the main package entry-point."""
    runner = CliRunner()
    result = runner.invoke(run)
    assert result.exit_code == 0
    assert "Entry point to start the tasks." in result.output
    assert "Options:" in result.output
    assert "Commands:" in result.output
