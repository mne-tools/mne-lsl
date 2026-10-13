from __future__ import annotations

from mne._fiff.meas_info import ContainsMixin, SetChannelsMixin
from mne.utils import check_version


def test_mne() -> None:
    """Test the evolution of the MNE-mixins."""
    methods = [elt for elt in dir(ContainsMixin) if elt[0] != "_"]
    assert methods == ["compensation_grade", "get_channel_types"]

    methods = [elt for elt in dir(SetChannelsMixin) if elt[0] != "_"]
    expected = [
        "anonymize",
        "get_montage",
        "plot_sensors",
        "rename_channels",
        "set_channel_types",
        "set_meas_date",
        "set_montage",
    ]
    if check_version("mne", "1.14"):
        expected.insert(5, "set_head_sphere")
    assert methods == expected
