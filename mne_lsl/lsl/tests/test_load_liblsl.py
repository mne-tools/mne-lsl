import pytest

from mne_lsl.lsl.load_liblsl import (
    _load_liblsl_environment_variables,
    _load_liblsl_wheel_path,
)


@pytest.fixture
def libpath():
    """Return the liblsl path bundled with mne-lsl."""
    try:
        return _load_liblsl_wheel_path()
    except RuntimeError:
        pytest.skip(
            "Could not load liblsl from the wheel path. Unsupportred platform.system()."
        )


def test_load_liblsl_environment_variables(monkeypatch, libpath):
    """Test loading liblsl from an environment variable."""
    assert _load_liblsl_environment_variables() is None
    monkeypatch.setenv("MNE_LSL_LIB", libpath)
    assert _load_liblsl_environment_variables() == libpath
    with pytest.warns(RuntimeWarning, match="outdated, use at your own discretion"):
        assert _load_liblsl_environment_variables(version_min=10101) == libpath
    monkeypatch.setenv("MNE_LSL_LIB", "non-existent")
    assert _load_liblsl_environment_variables() is None
