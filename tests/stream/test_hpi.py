from __future__ import annotations

import pytest

from mne_lsl.stream._hpi import check_hpi_ch_names_megin


def test_check_hpi_ch_names_megin() -> None:
    """Test the channel name validation."""
    with pytest.raises(RuntimeError, match="Expected HPI channel names"):
        check_hpi_ch_names_megin(["foo"])
