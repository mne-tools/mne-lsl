from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from mne_lsl.datasets import sample, testing
from mne_lsl.datasets.sample import _REGISTRY as _REGISTRY_SAMPLE
from mne_lsl.datasets.testing import _REGISTRY as _REGISTRY_TESTING
from mne_lsl.utils._tests import sha256sum

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.xfail(reason="Connection issue to the dataset servers.")
@pytest.mark.parametrize("dataset", [sample, testing])
def test_data_path(dataset) -> None:
    """Test download if the testing dataset."""
    if dataset != testing and os.getenv("GITHUB_ACTIONS", "") == "true":
        pytest.skip("Skip sample dataset download on GitHub Actions.")
    path = dataset.data_path()
    assert path.exists()


@pytest.mark.xfail(reason="Connection issue to the dataset servers.")
@pytest.mark.parametrize(
    ("dataset", "registry"), [(sample, _REGISTRY_SAMPLE), (testing, _REGISTRY_TESTING)]
)
def test_make_registry(tmp_path: Path, dataset, registry: Path) -> None:
    """Test the registry tmp_path making."""
    if dataset != testing and os.getenv("GITHUB_ACTIONS", "") == "true":
        pytest.skip("Skip sample dataset download on GitHub Actions.")
    dataset._make_registry(dataset.data_path(), output=tmp_path / "registry.txt")
    assert (tmp_path / "registry.txt").exists()
    assert sha256sum(tmp_path / "registry.txt") == sha256sum(registry)
