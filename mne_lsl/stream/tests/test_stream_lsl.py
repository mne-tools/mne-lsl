from __future__ import annotations  # c.f. PEP 563, PEP 649

import logging
import os
import platform
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import pyplot as plt
from mne import Info, create_info, pick_info, pick_types
from mne.channels import DigMontage
from mne.io import RawArray
from mne.utils import check_version
from numpy.testing import assert_allclose
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF
    from mne._fiff.pick import _picks_to_idx
else:
    from mne.io.constants import FIFF
    from mne.io.pick import _picks_to_idx

from mne_lsl.lsl import StreamInfo, StreamOutlet
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.utils._tests import match_stream_and_raw_data
from mne_lsl.utils.logs import _use_log_level

if TYPE_CHECKING:
    from mne.io import BaseRaw


bad_gh_macos = pytest.mark.skipif(
    platform.system() == "Darwin" and os.getenv("GITHUB_ACTIONS", "") == "true",
    reason="Unreliable on macOS CIs.",
)


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


@pytest.fixture(scope="function")
def mock_lsl_stream_int(request):
    """Create a mock LSL stream streaming the channel number continuously."""
    # nest the PlayerLSL import to first write the temporary LSL configuration file
    from mne_lsl.player import PlayerLSL  # noqa: E402

    info = create_info(5, 1000, "eeg")
    data = np.full((5, 1000), np.arange(5).reshape(-1, 1))
    raw = RawArray(data, info)

    with PlayerLSL(raw, name=f"P_{request.node.name}") as player:
        yield player


@pytest.fixture(scope="function")
def mock_lsl_stream_annotations(raw_annotations, request):
    """Create a mock LSL stream streaming the channel number continuously."""
    # nest the PlayerLSL import to first write the temporary LSL configuration file
    from mne_lsl.player import PlayerLSL  # noqa: E402

    with PlayerLSL(raw_annotations, name=f"P_{request.node.name}") as player:
        yield player


@pytest.fixture(scope="function")
def raw_sinusoids() -> BaseRaw:
    """Create a raw object with sinusoids."""
    times = np.arange(0, 2, 1 / 1000)
    data1 = np.sin(2 * np.pi * 10 * times) + np.sin(2 * np.pi * 30 * times)
    data2 = np.sin(2 * np.pi * 30 * times) + np.sin(2 * np.pi * 50 * times)
    data3 = np.sin(2 * np.pi * 30 * times) + np.sin(2 * np.pi * 100 * times)
    data = np.vstack([data1, data2, data3])
    info = create_info(
        ch_names=["10-30", "30-50", "30-100"], sfreq=1000, ch_types="eeg"
    )
    return RawArray(data, info)


@pytest.fixture(scope="function")
def mock_lsl_stream_sinusoids(raw_sinusoids, request):
    """Create a mock LSL stream streaming sinusoids."""
    # nest the PlayerLSL import to first write the temporary LSL configuration file
    from mne_lsl.player import PlayerLSL

    with PlayerLSL(raw_sinusoids, name=f"P_{request.node.name}") as player:
        yield player


def test_stream(mock_lsl_stream, acquisition_delay, raw):
    """Test a valid Stream."""
    # test connect/disconnect
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    assert stream._info is None
    assert not stream.connected
    stream.connect(acquisition_delay=acquisition_delay)
    assert isinstance(stream.info, Info)
    assert stream.connected
    stream.disconnect()
    assert stream._info is None
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
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
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
    # compensation grade
    assert stream.compensation_grade is None
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
    with pytest.raises(ValueError, match="must be a positive number"):
        Stream(bufsize=2).connect(acquisition_delay=-1)


def test_stream_connection_no_args(mock_lsl_stream):
    """Test connection to the only available stream."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    assert stream._info is None
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


@bad_gh_macos
def test_stream_drop_channels(mock_lsl_stream, acquisition_delay, raw):
    """Test dropping channels."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect(acquisition_delay=acquisition_delay)
    time.sleep(0.1)  # give a bit of time to the stream to acquire the first chunks
    stream.drop_channels("hEOG")
    raw_ = raw.copy().drop_channels("hEOG")
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
    stream.drop_channels(["F7", "Fp2"])
    raw_ = raw_.drop_channels(["F7", "Fp2"])
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)

    # test pick after drop
    stream.set_channel_types({"M1": "emg", "M2": "emg"})
    stream.pick("emg")
    raw_.set_channel_types({"M1": "emg", "M2": "emg"})
    raw_.pick("emg")
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)

    # test invalid channel names
    with pytest.raises(ValueError, match="must contain existing channel names."):
        stream.drop_channels(["101", "F6"])
    with pytest.raises(ValueError, match="must contain existing channel names."):
        stream.drop_channels("101")

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
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)

    # change channel types for testing and pick again
    stream.set_channel_types({"M1": "emg", "M2": "emg"})
    stream.pick("eeg")
    raw_.set_channel_types({"M1": "emg", "M2": "emg"})
    raw_.pick("eeg")
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)

    # test dropping channels after pick
    stream.drop_channels(["F1", "F2"])
    raw_.drop_channels(["F1", "F2"])
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)

    # test lack of re-order via pick
    stream.pick(
        [stream.ch_names[5], stream.ch_names[3], stream.ch_names[8], stream.ch_names[1]]
    )
    raw_.pick([raw_.ch_names[1], raw_.ch_names[3], raw_.ch_names[5], raw_.ch_names[8]])
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
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


@bad_gh_macos
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
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(stream._acquisition_delay, mock_lsl_stream)
    stream.disconnect()


def test_stream_channel_units(mock_lsl_stream, raw):
    """Test channel unit getters and setters."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect()
    time.sleep(0.1)
    ch_units = stream.get_channel_units()
    assert ch_units == [(FIFF.FIFF_UNIT_NONE, FIFF.FIFF_UNITM_NONE)] + [
        (FIFF.FIFF_UNIT_V, FIFF.FIFF_UNITM_NONE)
    ] * (len(stream.ch_names) - 1)
    stream.set_channel_units({"vEOG": "microvolts", "hEOG": "uv", "TRIGGER": 3})
    ch_units = stream.get_channel_units()
    assert ch_units[stream.ch_names.index("vEOG")][1] == -6
    assert ch_units[stream.ch_names.index("hEOG")][1] == -6
    assert ch_units[stream.ch_names.index("TRIGGER")][1] == 3

    # set channel units after channel selection
    stream.pick(["vEOG", "hEOG", "TRIGGER", "F7", "Fp2"])
    raw_ = raw.copy().pick(["Fp2", "F7", "vEOG", "hEOG", "TRIGGER"])
    stream.set_channel_units({"F7": -6, "vEOG": 6})
    ch_units = stream.get_channel_units()
    assert ch_units[stream.ch_names.index("F7")][1] == -6
    assert ch_units[stream.ch_names.index("vEOG")][1] == 6
    # acquire a couple of chunks
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(stream._acquisition_delay, mock_lsl_stream)
    stream.disconnect()


@bad_gh_macos
def test_stream_add_reference_channels(mock_lsl_stream, acquisition_delay, raw):
    """Test add reference channels and channel selection."""
    stream = Stream(bufsize=2, name=mock_lsl_stream.name)
    stream.connect(acquisition_delay=acquisition_delay)
    time.sleep(0.1)  # give a bit of time to slower CIs
    stream.add_reference_channels("CPz")
    raw_ = raw.copy().add_reference_channels("CPz")
    assert stream.ch_names == raw_.ch_names
    time.sleep(0.2)
    # acquire a couple of chunks
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
    stream.add_reference_channels(["Ref1", "Ref2"])
    raw_.add_reference_channels(["Ref1", "Ref2"])
    assert stream.ch_names == raw_.ch_names
    # acquire a couple of chunks
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
    # pick channels
    stream.pick("eeg")
    raw_.pick("eeg")
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
    with pytest.raises(RuntimeError, match="selection would not leave any channel"):
        stream.pick("CPz")
    # add reference channel again
    stream.add_reference_channels("Ref3")
    raw_.add_reference_channels("Ref3")
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1)
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
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
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1, picks="eeg")
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
    raw_.pick(["F7", "F2", "F4"])
    time.sleep(0.2)
    for _ in range(3):
        data, _ = stream.get_data(winsize=0.1, picks=["F7", "F2", "F4"])
        match_stream_and_raw_data(data, raw_)
        _sleep_until_new_data(acquisition_delay, mock_lsl_stream)
    stream.disconnect()


def test_stream_n_new_samples(mock_lsl_stream, caplog):
    """Test the number of new samples available."""
    stream = Stream(bufsize=0.4, name=mock_lsl_stream.name)
    assert stream._n_new_samples is None
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
    _sleep_until_new_data(acquisition_delay, mock_lsl_stream_int)
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
    time.sleep(0.1)
    stream.add_reference_channels("5")
    time.sleep(0.1)
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
    _sleep_until_new_data(stream._acquisition_delay, mock_lsl_stream_int)
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
    _sleep_until_new_data(stream._acquisition_delay, mock_lsl_stream_int)
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
    _sleep_until_new_data(stream._acquisition_delay, mock_lsl_stream_int)
    data, _ = stream.get_data(picks="eeg")
    assert_allclose(data, data_ref)
    stream.disconnect()


def _sleep_until_new_data(acq_delay, player):
    """Sleep until new data is available, majorated by 10%."""
    time.sleep(
        max(
            1.1 * acq_delay,
            1.1 * (player.chunk_size / player.info["sfreq"]),
        )
    )


def test_stream_str(close_io):
    """Test a stream on a string source."""
    sinfo = StreamInfo("test_stream_str", "gaze", 1, 100, "string", "pytest")
    outlet = StreamOutlet(sinfo)
    assert outlet.dtype == "string"
    with pytest.raises(
        RuntimeError, match="Stream class is designed for numerical types"
    ):
        Stream(bufsize=2, name="test_stream_str").connect()
    close_io()


def test_stream_processing_flags(close_io):
    """Test a stream connection processing flags."""
    sinfo = StreamInfo("test_stream_processing_flags", "gaze", 1, 100, "int8", "pytest")
    outlet = StreamOutlet(sinfo)
    assert outlet.dtype == np.int8
    stream = Stream(bufsize=2, name="test_stream_processing_flags")
    assert not stream.connected
    with pytest.raises(
        ValueError, match="'threadsafe' processing flag should not be provided"
    ):
        stream.connect(processing_flags=("clocksync", "threadsafe"))
    assert not stream.connected
    stream.connect(processing_flags="all")
    assert stream.connected
    stream.disconnect()
    assert not stream.connected
    close_io()


def test_stream_irregularly_sampled(close_io):
    """Test a stream with an irregular sampling rate."""
    sinfo = StreamInfo(
        "test_stream_irregularly_sampled", "gaze", 1, 0, "int8", "pytest"
    )
    outlet = StreamOutlet(sinfo)
    stream = Stream(bufsize=10, name="test_stream_irregularly_sampled")
    stream.connect()
    time.sleep(0.1)  # give a bit of time to the stream to acquire the first chunks
    assert stream.connected
    data, _ = stream.get_data()
    expected = np.zeros(stream.n_buffer, dtype=stream.dtype)
    assert_allclose(data.squeeze(), expected)
    outlet.push_sample(np.array([1]))
    time.sleep(0.01)
    data, _ = stream.get_data()
    expected[-1] = 1
    assert_allclose(data.squeeze(), expected)
    with pytest.raises(RuntimeError, match="with an irregular sampling rate."):
        stream._check_connected_and_regular_sampling("test")
    stream.disconnect()
    close_io()


def test_stream_annotations_picks(mock_lsl_stream_annotations):
    """Test sub-selection of annotations."""
    stream = (
        Stream(
            bufsize=5,
            name=f"{mock_lsl_stream_annotations.name}-annotations",
            stype="annotations",
        )
        .connect()
        .pick("test1")
    )
    time.sleep(5)  # acquire data
    data, ts = stream.get_data()
    assert np.count_nonzero(data) == data.size
    stream.disconnect()


def test_stream_filter_deletion(mock_lsl_stream, caplog):
    """Test deletion of filters applied to a Stream."""
    # test no filter
    stream = Stream(bufsize=2.0, name=mock_lsl_stream.name).connect()
    time.sleep(0.1)
    with pytest.raises(RuntimeError, match="No filter to remove."):
        stream.del_filter("all")
    with pytest.raises(RuntimeError, match="No filter to remove."):
        stream.del_filter(0)
    # test valid deletion
    stream.filter(1, 100, picks=["F7", "F3", "Fz"])
    time.sleep(0.1)
    assert len(stream.filters) == 1
    stream.del_filter("all")
    assert len(stream.filters) == 0
    stream.filter(1, 100, picks=["F7", "F3", "Fz"])
    time.sleep(0.1)
    # test invalid
    with pytest.raises(ValueError, match="is provided as str, it must be"):
        stream.del_filter("0")
    with pytest.raises(ValueError, match="is provided as str, it must be"):
        stream.del_filter("0")
    with pytest.raises(TypeError, match="must be an instance of int-like"):
        stream.del_filter(["0"])
    with pytest.raises(TypeError, match="must be an instance of int-like"):
        stream.del_filter(("0",))
    with pytest.raises(TypeError, match="must be an instance of"):
        stream.del_filter((lambda x: 0,))
    with pytest.raises(TypeError, match="must be an instance of"):
        stream.del_filter(lambda x: 0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        stream.del_filter(1)
    with pytest.warns(RuntimeWarning, match="contains duplicates"):
        stream.del_filter((0, 0))
    assert len(stream.filters) == 0
    # test reset of initial conditions
    stream.filter(1, 100, picks=["F7", "F3", "Fz"])
    stream.filter(20, None, picks=["F7", "F3", "O1"])
    stream.filter(None, 20, picks=["Fz", "O2"])
    assert len(stream.filters) == 3
    assert stream.filters[0]["l_freq"] == 1.0
    assert stream.filters[1]["l_freq"] == 20.0
    assert stream.filters[2]["l_freq"] is None
    time.sleep(0.5)
    assert all(filt["zi"] is not None for filt in stream.filters)
    caplog.set_level(logging.INFO)
    caplog.clear()
    stream.del_filter(2)
    assert (
        f"The initial conditions will be reset on filters:\n{stream.filters[0]}"
    ) in caplog.text
    assert repr(stream.filters[1]) not in caplog.text
    stream.disconnect()


def test_stream_filter(mock_lsl_stream_sinusoids, raw_sinusoids):
    """Test stream filters."""
    freqs = fftfreq(raw_sinusoids.times.size, 1 / raw_sinusoids.info["sfreq"])
    idx = np.where(0 <= freqs)[0]
    freqs = freqs[idx]
    fft_orig = np.abs(fft(raw_sinusoids.get_data(), axis=-1)[:, idx])
    # extract peaks
    assert fft_orig.shape[0] == len(raw_sinusoids.ch_names)
    assert fft_orig.shape[0] == len(mock_lsl_stream_sinusoids.ch_names)
    heights_orig = dict()
    for k in range(fft_orig.shape[0]):
        peaks, _ = find_peaks(fft_orig[k, :], height=100)  # peak height is 1000
        fqs = [int(elt) for elt in raw_sinusoids.ch_names[k].split("-")]
        assert_allclose(freqs[peaks], fqs, atol=0.1)
        heights_orig[k] = dict(idx=peaks, heights=fft_orig[k, peaks])
    # test unfiltered data
    stream = Stream(bufsize=2.0, name=mock_lsl_stream_sinusoids.name).connect()
    time.sleep(2.1)
    fft_ = np.abs(fft(stream.get_data()[0], axis=-1)[:, idx])
    for ch, ch_height in heights_orig.items():
        assert_allclose(fft_[ch, ch_height["idx"]], ch_height["heights"], rtol=0.05)
    # test filtering
    stream.filter(5, 15, picks="10-30")
    time.sleep(2.1)
    fft_ = np.abs(fft(stream.get_data()[0], axis=-1)[:, idx])
    for ch, ch_height in heights_orig.items():
        if ch == 0:  # 10 Hz retained, 30 Hz removed
            assert fft_[ch, ch_height["idx"]][1] < 0.1 * ch_height["heights"][1]
            assert_allclose(
                fft_[ch, ch_height["idx"]][0], ch_height["heights"][0], rtol=0.05
            )
        else:
            assert_allclose(fft_[ch, ch_height["idx"]], ch_height["heights"], rtol=0.05)
    # test removing filter
    stream.del_filter(0)
    time.sleep(2.1)
    fft_ = np.abs(fft(stream.get_data()[0], axis=-1)[:, idx])
    for ch, ch_height in heights_orig.items():
        assert_allclose(fft_[ch, ch_height["idx"]], ch_height["heights"], rtol=0.05)
    # test adding multiple filters
    stream.filter(20, 70, picks="eeg")
    time.sleep(2.1)
    fft_ = np.abs(fft(stream.get_data()[0], axis=-1)[:, idx])
    for ch, ch_height in heights_orig.items():
        if ch == 0:  # 10 Hz removed, 30 Hz retained
            assert fft_[ch, ch_height["idx"]][0] < 0.1 * ch_height["heights"][0]
            assert_allclose(
                fft_[ch, ch_height["idx"]][1], ch_height["heights"][1], rtol=0.05
            )
        elif ch == 1:  # 30 Hz retained, 50 Hz retained
            assert_allclose(fft_[ch, ch_height["idx"]], ch_height["heights"], rtol=0.05)
        elif ch == 2:  # 30 Hz retained, 100 Hz removed (but not as much attenuation)
            assert fft_[ch, ch_height["idx"]][1] < 0.15 * ch_height["heights"][1]
            assert_allclose(
                fft_[ch, ch_height["idx"]][0], ch_height["heights"][0], rtol=0.05
            )
    stream.filter(40, 60, picks="30-50")  # second filter
    time.sleep(2.1)
    fft_ = np.abs(fft(stream.get_data()[0], axis=-1)[:, idx])
    for ch, ch_height in heights_orig.items():
        if ch == 0:  # 10 Hz removed, 30 Hz retained
            assert fft_[ch, ch_height["idx"]][0] < 0.1 * ch_height["heights"][0]
            assert_allclose(
                fft_[ch, ch_height["idx"]][1], ch_height["heights"][1], rtol=0.05
            )
        elif ch == 1:  # 30 Hz removed, 50 Hz retained
            assert fft_[ch, ch_height["idx"]][0] < 0.1 * ch_height["heights"][0]
            assert_allclose(
                fft_[ch, ch_height["idx"]][1], ch_height["heights"][1], rtol=0.05
            )
        elif ch == 2:  # 30 Hz retained, 100 Hz removed
            assert_allclose(
                fft_[ch, ch_height["idx"]][0], ch_height["heights"][0], rtol=0.05
            )
            assert fft_[ch, ch_height["idx"]][1] < 0.15 * ch_height["heights"][1]
    stream.filter(40, 60, picks="eeg")  # third filter
    time.sleep(2.1)
    fft_ = np.abs(fft(stream.get_data()[0], axis=-1)[:, idx])
    for ch, ch_height in heights_orig.items():
        if ch == 0:  # 10 Hz removed, 30 Hz removed
            assert fft_[ch, ch_height["idx"]][0] < 0.1 * ch_height["heights"][0]
            assert fft_[ch, ch_height["idx"]][0] < 0.1 * ch_height["heights"][0]
        elif ch == 1:  # 30 Hz removed, 50 Hz retained
            assert fft_[ch, ch_height["idx"]][0] < 0.1 * ch_height["heights"][0]
            assert_allclose(
                fft_[ch, ch_height["idx"]][1], ch_height["heights"][1], rtol=0.05
            )
        elif ch == 2:  # 30 Hz removed, 100 Hz removed
            assert fft_[ch, ch_height["idx"]][0] < 0.1 * ch_height["heights"][0]
            assert fft_[ch, ch_height["idx"]][1] < 0.15 * ch_height["heights"][1]
    stream.disconnect()
