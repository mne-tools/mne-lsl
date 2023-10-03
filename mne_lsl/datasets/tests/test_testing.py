from importlib.resources import files

from mne_lsl.datasets.testing import _make_registry
from mne_lsl.utils._tests import sha256sum


def test_testing_registry_up_to_date(tmp_path):
    """Test that the registry for the testing dataset is up to date."""
    registry_fname = files("mne_lsl.datasets") / "testing-registry.txt"
    _make_registry(tmp_path / "testing-registry.txt")
    assert sha256sum(registry_fname) == sha256sum(tmp_path / "testing-registry.txt")
