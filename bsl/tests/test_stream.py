from matplotlib import pyplot as plt
from mne import Info
from mne.channels import DigMontage
from mne.io import read_raw

from bsl import Stream
from bsl.datasets import testing

fname = testing.data_path() / "sample-eeg-ant-raw.fif"
raw = read_raw(fname, preload=True)


def test_stream(mock_lsl_stream):
    """Test a valid Stream."""
    # test connect/disconnect
    stream = Stream(bufsize=2, name="BSL-Player-pytest")
    assert stream.info is None
    assert not stream.connected
    stream.connect()
    assert isinstance(stream.info, Info)
    assert stream.connected
    stream.disconnect()
    assert stream.info is None
    assert not stream.connected
    stream.connect()
    assert isinstance(stream.info, Info)
    assert stream.connected

    # test content
    assert stream.info["ch_names"] == raw.info["ch_names"]
    assert stream.get_channel_types() == raw.get_channel_types()

    # montage
    stream.set_montage("standard_1020")
    stream.plot_sensors()
    plt.close("all")
    montage = stream.get_montage()
    assert isinstance(montage, DigMontage)
    assert montage.ch_names == stream.ch_names[1:]  # first channel is TRIGGER
