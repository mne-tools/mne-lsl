import re
import time
from datetime import datetime, timezone

import numpy as np
import pytest
from matplotlib import pyplot as plt
from mne import Info, pick_info, pick_types
from mne.channels import DigMontage
from mne.utils import check_version
from numpy.testing import assert_allclose

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF
    from mne._fiff.pick import _picks_to_idx
else:
    from mne.io.constants import FIFF
    from mne.io.pick import _picks_to_idx

from mne_lsl import logger
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.utils._tests import match_stream_and_raw_data
from mne_lsl.utils.logs import _use_log_level

logger.propagate = True


@pytest.fixture(
    params=(
        pytest.param(0.001, id="1ms"),
        pytest.param(0.2, id="200ms"),
        pytest.param(1, id="1s", marks=pytest.mark.slow),
    ),
)
def acquisition_delay(request):
    """Yield the acquisition delay of the mock LSL stream."""
    yield request.param


def test_stream(mock_lsl_stream, acquisition_delay, raw):
    """Test a valid Stream."""
    # test connect/disconnect
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    assert stream.info is None
    assert not stream.connected
    stream.connect(acquisition_delay=acquisition_delay)
    assert isinstance(stream.info, Info)
    assert stream.connected
    stream.disconnect()
    assert stream.info is None
    assert not stream.connected
    stream.connect(acquisition_delay=acquisition_delay)
    assert isinstance(stream.info, Info)
    assert stream.connected
    # test content
    assert stream.info["ch_names"] == raw.info["ch_names"]
    assert stream.get_channel_types() == raw.get_channel_types()
    assert stream.info["sfreq"] == raw.info["sfreq"]
    # check fs and that the returned data array is in raw a couple of times
    time.sleep(0.1)  # give a bit of time to the stream to acquire the first chunks
    for _ in range(3):
        data, ts = stream.get_data(winsize=0.1)
        assert ts.size == data.shape[1]
        assert_allclose(1 / np.diff(ts), stream.info["sfreq"])
        match_stream_and_raw_data(data, raw)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    # montage
    stream.set_montage("standard_1020")
    stream.plot_sensors()
    plt.close("all")
    montage = stream.get_montage()
    assert isinstance(montage, DigMontage)
    assert (
        montage.ch_names
        == pick_info(stream.info, _picks_to_idx(stream.info, "eeg")).ch_names
    )
    # dtype
    assert stream.dtype == stream.sinfo.dtype
    # disconnect
    stream.disconnect()


def test_stream_invalid():
    """Test creation and connection to an invalid stream."""
    with pytest.raises(RuntimeError, match="do not uniquely identify an LSL stream"):
        stream = Stream(bufsize=2, name="101")
        stream.connect()
    with pytest.raises(RuntimeError, match="do not uniquely identify an LSL stream"):
        stream = Stream(bufsize=2, stype="EEG")
        stream.connect()
    with pytest.raises(RuntimeError, match="do not uniquely identify an LSL stream"):
        stream = Stream(bufsize=2, source_id="101")
        stream.connect()
    with pytest.raises(TypeError, match="must be an instance of numeric"):
        Stream(bufsize="101")
    with pytest.raises(ValueError, match="must be a strictly positive number"):
        Stream(bufsize=0)
    with pytest.raises(ValueError, match="must be a strictly positive number"):
        Stream(bufsize=-101)
    with pytest.raises(TypeError, match="must be an instance of str"):
        Stream(1, name=101)
    with pytest.raises(TypeError, match="must be an instance of str"):
        Stream(1, stype=101)
    with pytest.raises(TypeError, match="must be an instance of str"):
        Stream(1, source_id=101)


def test_stream_connection_no_args(mock_lsl_stream):
    """Test connection to the only available stream."""
    stream = Stream(bufsize=2)
    assert stream.info is None
    assert not stream.connected
    assert stream.name is None
    assert stream.stype is None
    assert stream.source_id is None
    stream.connect()
    assert isinstance(stream.info, Info)
    assert stream.connected
    assert stream.name == mock_lsl_stream.name
    assert stream.stype == ""
    assert stream.source_id == "MNE-LSL"
    stream.disconnect()


def test_stream_double_connection(mock_lsl_stream, caplog):
    """Test connecting twice to a stream."""
    caplog.set_level(30)  # WARNING
    caplog.clear()
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect()
    time.sleep(0.1)  # give a bit of time to the stream to acquire the first chunks
    assert "stream is already connected" not in caplog.text
    caplog.clear()
    stream.connect()
    assert "stream is already connected" in caplog.text
    stream.disconnect()


def test_stream_drop_channels(mock_lsl_stream, acquisition_delay, raw):
    """Test dropping channels."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect(acquisition_delay=acquisition_delay)
    time.sleep(0.1)  # give a bit of time to the stream to acquire the first chunks
    # Do not drop TRIGGER as it contains our increasing timestamps
    assert raw.ch_names[-1] == "TRIGGER"
    stream.drop_channels("hEOG")
    raw_ = raw.copy().drop_channels("hEOG")
    assert stream.ch_names == raw_.ch_names
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    stream.drop_channels(["Fp1", "Fp2"])
    raw_ = raw_.drop_channels(["Fp1", "Fp2"])
    assert stream.ch_names == raw_.ch_names
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)

    # test pick after drop
    stream.set_channel_types({"M1": "emg", "M2": "emg"})
    stream.pick("emg")
    raw_.set_channel_types({"M1": "emg", "M2": "emg"})
    raw_.pick("emg")
    assert stream.ch_names == raw_.ch_names
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    stream.disconnect()


def test_stream_pick(mock_lsl_stream, acquisition_delay, raw):
    """Test channel selection."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect(acquisition_delay=acquisition_delay)
    time.sleep(0.1)  # give a bit of time to the stream to acquire the first chunks
    stream.info["bads"] = ["Fp2"]
    stream.pick("eeg", exclude="bads")
    raw_ = raw.copy()
    raw_.info["bads"] = ["Fp2"]
    raw_.pick("eeg", exclude="bads")
    assert stream.ch_names == raw_.ch_names
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)

    # change channel types for testing and pick again
    stream.set_channel_types({"M1": "emg", "M2": "emg"})
    stream.pick("eeg")
    raw_.set_channel_types({"M1": "emg", "M2": "emg"})
    raw_.pick("eeg")
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.1)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)

    # test dropping channels after pick
    stream.drop_channels(["F1", "F2"])
    raw_.drop_channels(["F1", "F2"])
    assert stream.ch_names == raw_.ch_names
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)

    # test lack of re-order via pick
    stream.pick(
        [stream.ch_names[5], stream.ch_names[3], stream.ch_names[8], stream.ch_names[1]]
    )
    raw_.pick([raw_.ch_names[1], raw_.ch_names[3], raw_.ch_names[5], raw_.ch_names[8]])
    assert stream.ch_names == raw_.ch_names
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    stream.disconnect()


def test_stream_meas_date_and_anonymize(mock_lsl_stream):
    """Test stream measurement date."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect()
    assert stream.info["meas_date"] is None
    meas_date = datetime(2023, 1, 25, tzinfo=timezone.utc)
    stream.set_meas_date(meas_date)
    assert stream.info["meas_date"] == meas_date
    stream.info["experimenter"] = "Mathieu Scheltienne"
    stream.anonymize(daysback=10)
    assert stream.info["meas_date"] == datetime(2023, 1, 15, tzinfo=timezone.utc)
    assert stream.info["experimenter"] != "Mathieu Scheltienne"
    stream.disconnect()


def test_stream_channel_types(mock_lsl_stream, raw):
    """Test channel type getters and setters."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect()
    assert stream.get_channel_types(unique=True) == raw.get_channel_types(unique=True)
    assert stream.get_channel_types(unique=False) == raw.get_channel_types(unique=False)
    assert "eeg" in stream
    stream.set_channel_types({"M1": "emg", "M2": "emg"})
    raw_ = raw.copy().set_channel_types({"M1": "emg", "M2": "emg"})
    assert stream.get_channel_types(unique=True) == raw_.get_channel_types(unique=True)
    assert stream.get_channel_types() == raw_.get_channel_types()
    assert "emg" in stream
    stream.disconnect()


def test_stream_channel_names(mock_lsl_stream, raw):
    """Test channel renaming."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect()
    time.sleep(0.1)
    assert stream.ch_names == raw.ch_names
    assert stream.info["ch_names"] == raw.ch_names
    stream.rename_channels({"M1": "EMG1", "M2": "EMG2"})
    raw_ = raw.copy().rename_channels({"M1": "EMG1", "M2": "EMG2"})
    assert stream.ch_names == raw_.ch_names
    assert stream.info["ch_names"] == raw_.ch_names

    # rename after channel selection
    stream.drop_channels("vEOG")
    raw_.drop_channels("vEOG")
    stream.rename_channels({"hEOG": "EOG"})
    raw_.rename_channels({"hEOG": "EOG"})
    assert stream.ch_names == raw_.ch_names
    assert stream.info["ch_names"] == raw_.ch_names
    # acquire a couple of chunks
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(0.04)
    stream.disconnect()


def test_stream_channel_units(mock_lsl_stream, raw):
    """Test channel unit getters and setters."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect()
    time.sleep(0.1)
    ch_units = stream.get_channel_units()
    assert ch_units == [(FIFF.FIFF_UNIT_V, FIFF.FIFF_UNITM_NONE)] * len(stream.ch_names)
    stream.set_channel_units({"vEOG": "microvolts", "hEOG": "uv", "TRIGGER": 3})
    ch_units = stream.get_channel_units()
    assert ch_units[stream.ch_names.index("vEOG")][1] == -6
    assert ch_units[stream.ch_names.index("hEOG")][1] == -6
    assert ch_units[stream.ch_names.index("TRIGGER")][1] == 3

    # set channel units after channel selection
    stream.pick(["vEOG", "hEOG", "TRIGGER", "Fp1", "Fp2"])
    raw_ = raw.copy().pick(["Fp1", "Fp2", "vEOG", "hEOG", "TRIGGER"])
    stream.set_channel_units({"Fp1": -6, "vEOG": 6})
    ch_units = stream.get_channel_units()
    assert ch_units[stream.ch_names.index("Fp1")][1] == -6
    assert ch_units[stream.ch_names.index("vEOG")][1] == 6
    # acquire a couple of chunks
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(0.04)
    stream.disconnect()


def test_stream_add_reference_channels(mock_lsl_stream, acquisition_delay, raw):
    """Test add reference channels and channel selection."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect(acquisition_delay=acquisition_delay)
    time.sleep(0.1)  # give a bit of time to slower CIs
    stream.add_reference_channels("CPz")
    raw_ = raw.copy().add_reference_channels("CPz")
    assert stream.ch_names == raw_.ch_names
    # acquire a couple of chunks
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    stream.add_reference_channels(["Ref1", "Ref2"])
    raw_.add_reference_channels(["Ref1", "Ref2"])
    assert stream.ch_names == raw_.ch_names
    # acquire a couple of chunks
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    # pick channels
    stream.pick("eeg")
    raw_.pick("eeg")
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    with pytest.raises(RuntimeError, match="selection would not leave any channel"):
        stream.pick("CPz")
    # add reference channel again
    stream.add_reference_channels("Ref3")
    raw_.add_reference_channels("Ref3")
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    stream.disconnect()


def test_stream_repr(mock_lsl_stream):
    """Test the stream representation."""
    stream = Stream(bufsize=2)
    assert stream.__repr__() == "<Stream: OFF>"
    name = mock_lsl_stream.name
    stream = Stream(bufsize=2, name=name)
    assert stream.__repr__() == f"<Stream: OFF | {name} (source: unknown)>"
    stream.connect()
    assert stream.__repr__() == f"<Stream: ON | {name} (source: MNE-LSL)>"
    stream.disconnect()
    stream = Stream(bufsize=2, name=name, source_id="MNE-LSL")
    assert stream.__repr__() == f"<Stream: OFF | {name} (source: MNE-LSL)>"
    stream = Stream(bufsize=2, source_id="MNE-LSL")
    assert stream.__repr__() == "<Stream: OFF | (source: MNE-LSL>"


def test_stream_get_data_picks(mock_lsl_stream, acquisition_delay, raw):
    """Test channel sub-selection when getting data."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect(acquisition_delay=acquisition_delay)
    time.sleep(0.1)  # give a bit of time to slower CIs
    stream.add_reference_channels("CPz")
    raw_ = raw.copy().add_reference_channels("CPz")
    raw_.pick("eeg")
    # acquire a couple of chunks
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1, picks="eeg")
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    raw_.pick(["Fp1", "F2", "F4"])
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1, picks=["Fp1", "F2", "F4"])
        match_stream_and_raw_data(data, raw_)
        time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    stream.disconnect()


def test_stream_n_new_samples(mock_lsl_stream, caplog):
    """Test the number of new samples available."""
    stream = Stream(bufsize=0.4, name=mock_lsl_stream.name)
    assert stream.n_new_samples is None
    stream.connect()
    time.sleep(0.1)  # give a bit of time to slower CIs
    assert stream.n_new_samples > 0
    _, _ = stream.get_data()
    # Between the above call and this one, samples could come in...
    # but hopefully not many
    assert stream.n_new_samples < 100
    with _use_log_level("INFO"):
        caplog.set_level(20)  # INFO
        caplog.clear()
        time.sleep(0.8)
        assert "new samples exceeds the buffer size" in caplog.text
    _, _ = stream.get_data(winsize=0.1)
    assert stream.n_new_samples < 100
    stream.disconnect()


def test_stream_invalid_interrupt(mock_lsl_stream):
    """Test invalid acquisition interruption."""
    stream = Stream(bufsize=0.4, name=mock_lsl_stream.name)
    assert not stream.connected
    with pytest.raises(RuntimeError, match="requested but the stream is not connected"):
        with stream._interrupt_acquisition():
            pass


def test_stream_rereference(mock_lsl_stream_int, acquisition_delay):
    """Test re-referencing an EEG-like stream."""
    stream = Stream(bufsize=0.4, name=mock_lsl_stream_int.name)
    stream.connect(acquisition_delay=acquisition_delay)
    time.sleep(0.1)  # give a bit of time to slower CIs
    assert stream.n_new_samples > 0
    data, _ = stream.get_data()
    assert_allclose(data, np.full(data.shape, np.arange(5).reshape(-1, 1)))

    stream.set_eeg_reference("1")
    data, _ = stream.get_data()
    data_ref = np.full(data.shape, np.arange(data.shape[0]).reshape(-1, 1))
    data_ref -= data_ref[1, :]
    assert_allclose(data, data_ref)
    time.sleep(acquisition_delay + 0.1 * acquisition_delay)
    data, _ = stream.get_data()
    assert_allclose(data, data_ref)

    with pytest.raises(RuntimeError, match=re.escape("set_eeg_reference() can only")):
        stream.set_eeg_reference("2")
    with pytest.raises(
        RuntimeError, match=re.escape("add_reference_channels() can only")
    ):
        stream.add_reference_channels("101")
    with pytest.raises(RuntimeError, match="selection must be done before adding a"):
        stream.drop_channels("4")

    stream.disconnect()
    assert stream._ref_channels is None
    assert stream._ref_from is None
    time.sleep(0.05)  # give a bit of time to slower CIs

    stream.connect()
    time.sleep(0.1)  # give a bit of time to slower CIs
    stream.add_reference_channels("5")
    data, _ = stream.get_data()
    data_ref = np.full(data.shape, np.arange(data.shape[0]).reshape(-1, 1))
    data_ref[-1, :] = np.zeros(data.shape[1])
    assert_allclose(data, data_ref)
    stream.set_eeg_reference(("1", "2"))
    data, _ = stream.get_data()
    data_ref = np.full(
        data.shape, np.arange(data.shape[0]).reshape(-1, 1), dtype=stream.dtype
    )
    data_ref[-1, :] = np.zeros(data.shape[1])
    data_ref -= data_ref[[1, 2], :].mean(axis=0, keepdims=True)
    assert_allclose(data, data_ref)
    time.sleep(0.04)
    data, _ = stream.get_data()
    assert_allclose(data, data_ref)
    stream.disconnect()


def test_stream_rereference_average(mock_lsl_stream_int):
    """Test average re-referencing schema."""
    stream = Stream(bufsize=0.4, name=mock_lsl_stream_int.name)
    stream.connect()
    time.sleep(0.1)  # give a bit of time to slower CIs
    stream.set_channel_types({"2": "ecg"})  # channels: 0, 1, 2, 3, 4
    data, _ = stream.get_data(picks="eeg")
    picks = pick_types(stream.info, eeg=True)
    data_ref = np.full(
        (picks.size, data.shape[1]), np.arange(picks.size).reshape(-1, 1)
    )
    data_ref[-2:, :] += 1
    assert_allclose(data, data_ref)
    time.sleep(0.04)
    data, _ = stream.get_data(picks="eeg")
    assert_allclose(data, data_ref)

    stream.set_eeg_reference("average")
    data, _ = stream.get_data(picks="eeg")
    data_ref = np.full(
        (picks.size, data.shape[1]),
        np.arange(picks.size).reshape(-1, 1),
        dtype=stream.dtype,
    )
    data_ref[-2:, :] += 1
    data_ref -= data_ref.mean(axis=0, keepdims=True)
    assert_allclose(data, data_ref)
    time.sleep(0.04)
    data, _ = stream.get_data(picks="eeg")
    assert_allclose(data, data_ref)
    stream.disconnect()
