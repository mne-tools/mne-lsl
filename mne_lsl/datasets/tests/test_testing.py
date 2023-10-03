from importlib.resources import files

from bsl.datasets.testing import _make_registry
from bsl.utils._tests import sha256sum


def test_testing_registry_up_to_date(tmp_path):
    """Test that the registry for the testing dataset is up to date."""
    registry_fname = files("bsl.datasets") / "testing-registry.txt"
    _make_registry(tmp_path / "testing-registry.txt")
    assert sha256sum(registry_fname) == sha256sum(tmp_path / "testing-registry.txt")
