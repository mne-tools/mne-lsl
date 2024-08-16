from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pooch
import pytest

from mne_lsl.datasets._fetch import fetch_dataset

if TYPE_CHECKING:
    from typing import Optional


@pytest.fixture
def license_file() -> Optional[Path]:
    """Find the license file if present."""
    fname = Path(__file__).parent.parent.parent.parent / "LICENSE"
    if fname.exists():
        return fname
    else:
        pytest.skip("License file not found.")


@pytest.fixture
def license_url() -> str:
    """Return the URL for the license file."""
    return "https://raw.githubusercontent.com/mne-tools/mne-lsl/main/"


@pytest.mark.xfail(reason="Connection issue.")
def test_fetch_dataset(tmp_path, license_file, license_url):
    """Test dataset fetching."""
    license_hash = pooch.file_hash(license_file)
    with open(tmp_path / "registry.txt", "w") as fid:
        fid.write(f"{license_file.name} {license_hash}\n")
    fetch_dataset(tmp_path, license_url, tmp_path / "registry.txt")
    assert (tmp_path / license_file.name).exists()
    license_hash_downloaded = pooch.file_hash(tmp_path / license_file.name)
    assert license_hash_downloaded == license_hash
