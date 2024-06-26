import pytest
from click.testing import CliRunner

from ..sys_info import run


@pytest.mark.parametrize("developer", [False, True])
def test_sys_info(developer: bool):
    """Test the system information entry-point."""
    runner = CliRunner()
    result = runner.invoke(run, ["--developer"] if developer else [])
    assert result.exit_code == 0
    assert "Platform:" in result.output
    assert "Python:" in result.output
    assert "Executable:" in result.output
    assert "Core dependencies" in result.output
    if developer:
        assert "Optional 'build' dependencies" in result.output
        assert "Optional 'style' dependencies" in result.output
        assert "Optional 'test' dependencies" in result.output
