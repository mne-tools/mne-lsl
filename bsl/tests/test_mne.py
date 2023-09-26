import pytest
from mne.utils import check_version


@pytest.mark.skipif(not check_version("mne", "1.6"), reason="requires MNE 1.6 or above")
def test_mne():
    """Test the evolution of the MNE-mixins."""
    from mne._fiff.meas_info import ContainsMixin, SetChannelsMixin

    methods = [elt for elt in dir(ContainsMixin) if elt[0] != "_"]
    assert methods == ["compensation_grade", "get_channel_types"]

    methods = [elt for elt in dir(SetChannelsMixin) if elt[0] != "_"]
    assert methods == [
        "anonymize",
        "get_montage",
        "plot_sensors",
        "rename_channels",
        "set_channel_types",
        "set_meas_date",
        "set_montage",
    ]
