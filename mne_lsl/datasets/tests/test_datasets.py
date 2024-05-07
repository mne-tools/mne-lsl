import pytest

from mne_lsl.datasets import sample, testing
from mne_lsl.datasets.sample import _REGISTRY as _REGISTRY_SAMPLE
from mne_lsl.datasets.testing import _REGISTRY as _REGISTRY_TESTING
from mne_lsl.utils._tests import sha256sum


@pytest.mark.parametrize("dataset", [sample, testing])
def test_data_path(dataset):
    """Test download if the testing dataset."""
    path = dataset.data_path()
    assert path.exists()


@pytest.mark.parametrize(
    ("dataset", "registry"), [(sample, _REGISTRY_SAMPLE), (testing, _REGISTRY_TESTING)]
)
def test_make_registry(tmp_path, dataset, registry):
    """Test the registrytmp_path making."""
    dataset._make_registry(dataset.data_path(), output=tmp_path / "registry.txt")
    assert (tmp_path / "registry.txt").exists()
    assert sha256sum(tmp_path / "registry.txt") == sha256sum(registry)
