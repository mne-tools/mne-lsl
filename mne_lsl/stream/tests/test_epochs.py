from __future__ import annotations

import multiprocessing as mp
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import annotations_from_events, create_info, find_events
from mne.io import RawArray, read_raw_fif
from mne.io.base import BaseRaw
from numpy.testing import assert_allclose

from mne_lsl.datasets import testing
from mne_lsl.stream import EpochsStream, StreamLSL
from mne_lsl.stream.epochs import (
    _check_baseline,
    _check_reject_flat,
    _check_reject_tmin_tmax,
    _ensure_detrend_str,
    _ensure_event_id,
    _find_events_in_stim_channels,
    _process_data,
    _prune_events,
    _remove_empty_elements,
)

if TYPE_CHECKING:
    from mne.io import BaseRaw
    from numpy.typing import NDArray


def test_ensure_event_id():
    """Test validation of event dictionary."""
    assert _ensure_event_id(5, None) == {"5": 5}
    assert _ensure_event_id({"5": 5}, None) == {"5": 5}

    with pytest.raises(ValueError, match="must be a positive integer or a dictionary"):
        _ensure_event_id(0, None)
    with pytest.raises(ValueError, match="must be a positive integer or a dictionary"):
        _ensure_event_id(-101, None)
    with pytest.raises(ValueError, match="must be a positive integer or a dictionary"):
        _ensure_event_id({"5": 0}, None)

    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id(5.5, None)
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id({"5": 5.5}, None)
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id({"101": None}, None)


def test_check_baseline():
    """Test validation of the baseline."""
    _check_baseline(None, -0.2, 0.5)
    _check_baseline((None, None), -0.2, 0.5)
    _check_baseline((None, 0), -0.2, 0.5)
    _check_baseline((0, None), -0.2, 0.5)
    _check_baseline((0, 0), -0.2, 0.5)

    with pytest.raises(ValueError, match="The beginning of the baseline period must"):
        _check_baseline((-0.2, 0), -0.1, 0.5)
    with pytest.raises(ValueError, match="The end of the baseline period must"):
        _check_baseline((-0.2, 0.8), -0.2, 0.6)

    with pytest.raises(TypeError, match="must be an instance of"):
        _check_baseline((-0.2, "test"), -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_baseline(("test", 0.5), -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_baseline(("test", "test"), -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_baseline(101, -0.2, 0.5)


@pytest.fixture(scope="session")
def info():
    """A simple info object with 10 EEG channels."""
    return create_info(
        ch_names=[f"ch{i}" for i in range(10)], sfreq=100, ch_types="eeg"
    )


def test_check_reject_flat(info):
    """Test validation of the rejection dictionaries."""
    _check_reject_flat(None, None, info)
    _check_reject_flat(None, {"eeg": 1e-6}, info)
    _check_reject_flat({"eeg": 1e-6}, None, info)
    _check_reject_flat({"eeg": 1e-6}, {"eeg": 1e-6}, info)

    with pytest.raises(ValueError, match="peak-to-peak rejection value"):
        _check_reject_flat({"eeg": -1e-6}, None, info)
    with pytest.raises(ValueError, match="flat rejection value"):
        _check_reject_flat(None, {"eeg": -1e-6}, info)
    with pytest.raises(ValueError, match="channel type '.*' in the .* is not part"):
        _check_reject_flat(None, {"eog": 1e-6}, info)
    with pytest.raises(ValueError, match="channel type '.*' in the .* is not part"):
        _check_reject_flat({"eog": 1e-6}, None, info)

    with pytest.raises(TypeError, match="must be an instance of"):
        _check_reject_flat(101, None, info)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_reject_flat(None, 101, info)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_reject_flat({"eeg": "test"}, None, info)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_reject_flat(None, {"eeg": "test"}, info)


def test_check_reject_tmin_tmax():
    """Test validation of rejection time windows."""
    _check_reject_tmin_tmax(None, None, -0.2, 0.5)
    _check_reject_tmin_tmax(-0.2, 0.5, -0.2, 0.5)
    _check_reject_tmin_tmax(None, 0.5, -0.2, 0.5)
    _check_reject_tmin_tmax(-0.2, None, -0.2, 0.5)
    _check_reject_tmin_tmax(0, 0.1, -0.2, 0.5)

    with pytest.raises(ValueError, match="The beginning of the rejection time window"):
        _check_reject_tmin_tmax(-0.5, 0.5, -0.2, 0.5)
    with pytest.raises(ValueError, match="end of the epoch period"):
        _check_reject_tmin_tmax(-0.2, 0.8, -0.2, 0.5)
    with pytest.raises(ValueError, match="must be greater than the beginning"):
        _check_reject_tmin_tmax(0.5, -0.2, -0.2, 0.5)

    with pytest.raises(TypeError, match="must be an instance of"):
        _check_reject_tmin_tmax("test", None, -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_reject_tmin_tmax(None, "test", -0.2, 0.5)


def test_ensure_detrend_str():
    """Test validation of detrend."""
    assert _ensure_detrend_str(None) is None
    assert _ensure_detrend_str(0) == "constant"
    assert _ensure_detrend_str(1) == "linear"
    assert _ensure_detrend_str("constant") == "constant"
    assert _ensure_detrend_str("linear") == "linear"

    with pytest.raises(ValueError, match="Invalid value for the 'detrend' parameter"):
        _ensure_detrend_str("test")

    with pytest.raises(TypeError, match="must be an integer"):
        _ensure_detrend_str(5.5)


@pytest.fixture()
def stim_channels_events() -> NDArray[np.int64]:
    """An event array that will be added to a set of stimulation channels."""
    return np.array(
        [
            [5, 0, 3],
            [10, 0, 1],
            [25, 0, 1],
            [30, 0, 2],
            [50, 0, 1],
            [70, 0, 1],
            [85, 0, 2],
        ],
        dtype=np.int64,
    )


@pytest.fixture()
def stim_channels(stim_channels_events: NDArray[np.int64]) -> NDArray[np.float64]:
    """A set of stimulation channels of shape (n_channels, n_samples)."""
    channels = np.zeros((2, 100))
    for k, ev in enumerate(stim_channels_events):
        idx = 0 if k not in (0, 2) else 1
        channels[idx, ev[0] : ev[0] + 10] = ev[2]
    return channels


def test_find_events_in_stim_channels(
    stim_channels_events: NDArray[np.int64],
    stim_channels: NDArray[np.float64],
):
    """Test finding events in stimulation channels."""
    events = _find_events_in_stim_channels(stim_channels, ["a", "b"], 100)
    assert_allclose(events, stim_channels_events)
    events = _find_events_in_stim_channels(
        stim_channels, ["a", "b"], 100, min_duration=0.2
    )
    assert events.size == 0
    with pytest.warns(RuntimeWarning, match="You have .* events shorter"):
        events = _find_events_in_stim_channels(
            stim_channels, ["a", "b"], 100, shortest_event=25
        )


@pytest.fixture()
def events() -> NDArray[np.int64]:
    """Return a simple event array.

    An event is present every 10 samples, cycling between the values (1, 2, 3).
    """
    n_events = 10
    events = np.zeros((n_events, 3), dtype=np.int64)
    for k in range(events.shape[0]):
        events[k, :] = [10 * (k + 1), 0, k % 3 + 1]
    return events


def test_prune_events(events: NDArray[np.int64]):
    """Test pruning events."""
    ts = np.arange(10000, 11000, 1.8)
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, None, None, 0)
    assert_allclose(events_, events)
    # test pruning events outside of the event_id dictionary
    events_ = _prune_events(events, dict(a=1, c=3), 10, ts, None, None, 0)
    assert sorted(np.unique(events_[:, 2])) == [1, 3]
    # test pruning events that can't fit in the buffer
    ts = np.arange(5)
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, None, None, 0)
    assert events_.size == 0
    ts = np.arange(10000, 11000, 1.8)  # ts.size == 556
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 500, ts, None, None, 0)
    assert events_[-1, 0] + 500 <= ts.size
    assert events_[-1, 0] == 50  # events @ 60, 70, 80, ... should be dropped
    # test fitting in the buffer with tmin
    ts = np.arange(15)
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, None, None, -7)
    assert events_.shape[0] == 1
    # test pruning events that have already been moved to the buffer
    ts = np.arange(10000, 11000, 1.8)  # ts.size == 556
    events_ = _prune_events(
        events, dict(a=1, b=2, c=3), 10, ts, ts[events[3, 0]], None, 0
    )
    assert_allclose(events_, events[4:, :])
    # test pruning events from an event stream, which converts the index to index in ts
    ts = np.arange(1000)
    ts_events = np.arange(500) * 2 + 0.5  # mock a different sampling frequency
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, None, ts_events, 0)
    assert_allclose(events_[:, 2], events[:, 2])
    # with the half sampling rate + 0.5 set above, we should be selecting:
    # from: 10, 20, 30, 40, ... corresponding to 20.5, 40.5, 60.5, ...
    # to: 21, 41, 61, ... corresponding to 20, 40, 60, ...
    assert_allclose(events_[:, 0], np.arange(20, 20 * (events_[:, 0].size + 1), 20) + 1)


@pytest.fixture()
def raw_with_stim_channel() -> BaseRaw:
    """Create a raw object with a stimulation channel.

    The raw object contains 1000 samples @ 1 kHz -> 1 second of data:
    - channel 0: index of the sample within the raw object
    - channel 1 and 2: 0, except when there is an event, then 101 during 100 samples
    - channel 3: 0, except when there is an event, then 1 during 10 samples (stim)

    There are 3 events @ 100, 500, 700 samples.
    """
    n_samples = 1000
    data = np.zeros((4, n_samples), dtype=np.float32)
    data[0, :] = np.arange(n_samples)  # index of the sample within the raw object
    for pos in (100, 500, 700):
        data[1:-1, pos : pos + 100] = 101
        data[-1, pos : pos + 10] = 1  # trigger channel at the end
    info = create_info(["ch0", "ch1", "ch2", "trg"], 1000, ["eeg"] * 3 + ["stim"])
    return RawArray(data, info)


def _player_mock_lsl_stream(
    raw: BaseRaw,
    name: str,
    chunk_size: int,
    status: mp.managers.ValueProxy,
) -> None:
    """Player for the 'mock_lsl_stream' fixture(s)."""
    # nest the PlayerLSL import to first write the temporary LSL configuration file
    from mne_lsl.player import PlayerLSL

    player = PlayerLSL(raw, chunk_size=chunk_size, name=name)
    player.start()
    status.value = 1
    while status.value:
        time.sleep(0.1)
    player.stop()


@pytest.fixture()
def _mock_lsl_stream(raw_with_stim_channel, request, chunk_size):
    """Create a mock LSL stream streaming events on a stim channel."""
    manager = mp.Manager()
    status = manager.Value("i", 0)
    name = f"P_{request.node.name}"
    process = mp.Process(
        target=_player_mock_lsl_stream,
        args=(raw_with_stim_channel, name, chunk_size, status),
    )
    process.start()
    while status.value != 1:
        pass
    yield
    status.value = 0
    process.join(timeout=2)
    process.kill()


@pytest.mark.usefixtures("_mock_lsl_stream")
def test_epochs_without_event_stream():
    """Test creating epochs from the main stream."""
    stream = StreamLSL(0.5).connect(acquisition_delay=0.1)
    epochs = EpochsStream(
        stream,
        10,
        event_channels="trg",
        event_id=dict(a=1),
        tmin=0,
        tmax=0.1,
        baseline=None,
    ).connect(acquisition_delay=0.1)
    while epochs.n_new_epochs == 0:
        time.sleep(0.1)
    n = epochs.n_new_epochs
    data = epochs.get_data()
    assert_allclose(data[:-n, :, :], np.zeros((10 - n, data.shape[1], data.shape[2])))
    data_channels = data[-n:, 1:-1, :]
    assert_allclose(data_channels, np.ones(data_channels.shape) * 101)
    # acquire more epochs
    while epochs.n_new_epochs < 3:
        time.sleep(0.1)
    n += epochs.n_new_epochs
    data = epochs.get_data()
    assert_allclose(data[:-n, :, :], np.zeros((10 - n, data.shape[1], data.shape[2])))
    data_channels = data[-n:, 1:-1, :]
    assert_allclose(data_channels, np.ones(data_channels.shape) * 101)
    epochs.disconnect()
    stream.disconnect()


@pytest.mark.usefixtures("_mock_lsl_stream")
def test_epochs_without_event_stream_tmin_tmax():
    """Test creating epochs from the main stream."""
    stream = StreamLSL(0.5).connect(acquisition_delay=0.1)
    epochs = EpochsStream(
        stream, 10, event_channels="trg", event_id=dict(a=1), tmin=-0.05, tmax=0.15
    ).connect(acquisition_delay=0.1)
    while epochs.n_new_epochs == 0:
        time.sleep(0.1)
    n = epochs.n_new_epochs
    data = epochs.get_data()
    assert_allclose(data[:-n, :, :], np.zeros((10 - n, data.shape[1], data.shape[2])))
    data_channels = data[-n:, 1:-1, :]
    assert_allclose(
        data_channels[:, :, :50],
        np.zeros((data_channels.shape[0], data_channels.shape[1], 50)),
    )
    assert_allclose(
        data_channels[:, :, 50:150],
        np.ones((data_channels.shape[0], data_channels.shape[1], 100)) * 101,
    )
    assert_allclose(
        data_channels[:, :, 150:],
        np.zeros((data_channels.shape[0], data_channels.shape[1], 50)),
    )
    epochs.disconnect()
    stream.disconnect()


@pytest.mark.usefixtures("_mock_lsl_stream")
def test_epochs_without_event_stream_manual_acquisition():
    """Test creating epochs from the main stream."""
    stream = StreamLSL(0.5).connect(acquisition_delay=0.1)
    epochs = EpochsStream(
        stream,
        10,
        event_channels="trg",
        event_id=dict(a=1),
        tmin=-0.05,
        tmax=0.15,
        baseline=(None, 0),
    ).connect(acquisition_delay=0)
    assert epochs.n_new_epochs == 0
    time.sleep(0.5)
    assert epochs.n_new_epochs == 0
    data = epochs.get_data()
    assert_allclose(data[:, :, :], np.zeros(data.shape))
    while epochs.n_new_epochs == 0:
        epochs.acquire()
        time.sleep(0.1)
    n = epochs.n_new_epochs
    data = epochs.get_data()
    assert epochs.n_new_epochs == 0
    data_channels = data[-n:, 1:-1, :]
    assert_allclose(
        data_channels[:, :, :50],
        np.zeros((data_channels.shape[0], data_channels.shape[1], 50)),
    )
    assert_allclose(
        data_channels[:, :, 50:150],
        np.ones((data_channels.shape[0], data_channels.shape[1], 100)) * 101,
    )
    assert_allclose(
        data_channels[:, :, 150:],
        np.zeros((data_channels.shape[0], data_channels.shape[1], 50)),
    )
    epochs.disconnect()
    stream.disconnect()


@pytest.fixture()
def data_ones() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Data array used for baseline correction test."""
    data = np.ones((2, 100, 5), dtype=np.float64)
    times = np.arange(100, dtype=np.float64)
    return data, times


def test_process_data_no_baseline(
    data_ones: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test processing data without baseline correction."""
    data = _process_data(
        data_ones[0],
        baseline=None,
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_ones[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data, data_ones[0])


def test_process_data_baseline_all_segment(
    data_ones: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test processing data with baseline correction on the entire segment."""
    data = _process_data(
        data_ones[0].copy(),
        baseline=(None, None),
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_ones[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data, np.zeros(data.shape))


def test_process_data_baseline_start_segment(
    data_ones: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test processing data with baseline correction on the start segment."""
    data_ones[0][:, 10:, :] = 101
    data = _process_data(
        data_ones[0].copy(),
        baseline=(None, 10),
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_ones[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data[:, :10, :], np.zeros((2, 10, 5)))
    assert_allclose(data[:, 10:, :], np.ones((2, 90, 5)) * 100)


def test_process_data_baseline_end_segment(
    data_ones: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test processing data with baseline correction on the end segment."""
    data_ones[0][:, :90, :] = 101
    data = _process_data(
        data_ones[0].copy(),
        baseline=(90, None),
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_ones[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data[:, 90:, :], np.zeros((2, 10, 5)))
    assert_allclose(data[:, :90, :], np.ones((2, 90, 5)) * 100)


def test_process_data_detrend_constant(
    data_ones: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test constant (DC) detrending."""
    data = _process_data(
        data_ones[0],
        baseline=None,
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_ones[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data, data_ones[0])

    data = _process_data(
        data_ones[0],
        baseline=None,
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type="constant",
        times=data_ones[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data, np.zeros(data.shape))


@pytest.fixture()
def data_detrend_linear() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Data array used for detrending test."""
    data = [
        np.arange(100 * k, 100 * (k + 1), dtype=np.float64).reshape(1, 100, 1)
        for k in range(3)
    ]
    data = np.concatenate(data, axis=-1)
    data = np.concatenate([data, data * 2], axis=0)
    times = np.arange(100, dtype=np.float64)
    return data, times


def test_process_data_detrend_linear(
    data_detrend_linear: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test linear detrending."""
    data = _process_data(
        data_detrend_linear[0],
        baseline=None,
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_detrend_linear[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data, data_detrend_linear[0])

    data = _process_data(
        data_detrend_linear[0],
        baseline=None,
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type="linear",
        times=data_detrend_linear[1],
        ch_idx_by_type=dict(),
    )
    assert_allclose(data, np.zeros(data.shape), atol=1e-6)


def test_process_data_flat(data_ones: tuple[NDArray[np.float64], NDArray[np.float64]]):
    """Test rejection of epochs due to flatness."""
    data = _process_data(
        data_ones[0],
        baseline=None,
        reject=None,
        flat=dict(eeg=1),
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_ones[1],
        ch_idx_by_type=dict(eeg=[0, 1]),
    )
    assert data.shape[0] == 0


@pytest.fixture()
def data_reject():
    """Data array used for rejection test."""
    data = np.ones((2, 100, 5), dtype=np.float64)
    data[0, ::2, :] = 2
    data[1, ::2, :] = 101
    times = np.arange(100, dtype=np.float64)
    return data, times


def test_process_data_reject(
    data_reject: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test rejection of epochs due to PTP."""
    assert data_reject[0].shape[0] == 2
    data = _process_data(
        data_reject[0].copy(),
        baseline=None,
        reject=dict(eeg=50),
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_reject[1],
        ch_idx_by_type=dict(eeg=[0, 1]),
    )
    assert data.shape[0] == 1

    data = _process_data(
        data_reject[0].copy(),
        baseline=None,
        reject=dict(eeg=500),
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_reject[1],
        ch_idx_by_type=dict(eeg=[0, 1]),
    )
    assert data.shape[0] == 2

    data = _process_data(
        data_reject[0].copy(),
        baseline=None,
        reject=dict(eeg=1e-3),
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        detrend_type=None,
        times=data_reject[1],
        ch_idx_by_type=dict(eeg=[0, 1]),
    )
    assert data.shape[0] == 0


@pytest.fixture()
def data_reject_tmin_tmax():
    """Data array used for rejection based on segment test."""
    data = np.ones((2, 100, 5), dtype=np.float64)
    data[0, 10::2, :] = 2
    data[1, 10::2, :] = 101
    times = np.arange(100, dtype=np.float64)
    return data, times


def test_process_data_reject_tmin_tmax(
    data_reject_tmin_tmax: tuple[NDArray[np.float64], NDArray[np.float64]],
):
    """Test rejection of epochs due to PTP during segment."""
    assert data_reject_tmin_tmax[0].shape[0] == 2
    data = _process_data(
        data_reject_tmin_tmax[0].copy(),
        baseline=None,
        reject=dict(eeg=50),
        flat=None,
        reject_tmin=0,
        reject_tmax=10,
        detrend_type=None,
        times=data_reject_tmin_tmax[1],
        ch_idx_by_type=dict(eeg=[0, 1]),
    )
    assert data.shape[0] == 2

    assert data_reject_tmin_tmax[0].shape[0] == 2
    data = _process_data(
        data_reject_tmin_tmax[0].copy(),
        baseline=None,
        reject=dict(eeg=50),
        flat=None,
        reject_tmin=10,
        reject_tmax=25,
        detrend_type=None,
        times=data_reject_tmin_tmax[1],
        ch_idx_by_type=dict(eeg=[0, 1]),
    )
    assert data.shape[0] == 1


@pytest.fixture()
def raw_with_annotations(raw_with_stim_channel: BaseRaw) -> BaseRaw:
    """Create a raw object with annotations instead of a stim channel.

    The raw object contains 1000 samples @ 1 kHz -> 1 second of data:
    - channel 0: index of the sample within the raw object
    - channel 1 and 2: 0, except when there is an event, then 101 during 100 samples
    - channel 3: 0, except when there is an event, then 1 during 10 samples (stim)

    Channel 3 is dropped after annotations are created from events.
    There are 3 events @ 100, 500, 700 samples.
    """
    events = find_events(raw_with_stim_channel, "trg")
    annotations = annotations_from_events(
        events,
        raw_with_stim_channel.info["sfreq"],
        event_desc={1: "event"},
        first_samp=raw_with_stim_channel.first_samp,
    )
    annotations.duration += 0.1
    return raw_with_stim_channel.drop_channels("trg").set_annotations(annotations)


@pytest.fixture()
def _mock_lsl_stream_with_annotations(raw_with_annotations, request, chunk_size):
    """Create a mock LSL stream streaming events with annotations."""
    manager = mp.Manager()
    status = manager.Value("i", 0)
    name = f"P_{request.node.name}"
    process = mp.Process(
        target=_player_mock_lsl_stream,
        args=(raw_with_annotations, name, chunk_size, status),
    )
    process.start()
    while status.value != 1:
        pass
    yield
    status.value = 0
    process.join(timeout=2)
    process.kill()


@pytest.mark.usefixtures("_mock_lsl_stream_with_annotations")
def test_epochs_with_irregular_numerical_event_stream():
    """Test creating epochs from an irregularly sampled numerical event stream."""
    event_stream = StreamLSL(10, stype="annotations").connect(acquisition_delay=0.1)
    name = event_stream.name.removesuffix("-annotations")
    stream = StreamLSL(0.5, name=name).connect(acquisition_delay=0.1)
    epochs = EpochsStream(
        stream,
        10,
        event_channels="event",
        event_stream=event_stream,
        event_id=None,
        tmin=0,
        tmax=0.1,
        baseline=None,
    ).connect(acquisition_delay=0.1)
    while epochs.n_new_epochs == 0:
        time.sleep(0.1)
    n = epochs.n_new_epochs
    data = epochs.get_data()
    assert_allclose(data[:-n, :, :], np.zeros((10 - n, data.shape[1], data.shape[2])))
    data_channels = data[-n:, 1:-1, 2:-2]  # give 2 sample of jitter
    assert_allclose(data_channels, np.ones(data_channels.shape) * 101)
    epochs.disconnect()
    stream.disconnect()
    event_stream.disconnect()


@pytest.mark.usefixtures("_mock_lsl_stream_with_annotations")
def test_ensure_event_id_with_event_stream():
    """Test validation of event dictionary when an event_stream is present."""
    with pytest.raises(ValueError, match="must be provided if no irregularly sampled"):
        _ensure_event_id(None, None)
    event_stream = StreamLSL(10, stype="annotations").connect(acquisition_delay=0.1)
    assert _ensure_event_id(None, event_stream) is None
    with pytest.warns(RuntimeWarning, match="should be set to None"):
        _ensure_event_id(dict(event=1), event_stream)
    event_stream.disconnect()


def test_remove_empty_elements():
    """Test _remove_empty_elements."""
    data = np.ones(10).reshape(1, -1)
    ts = np.zeros(10)
    ts[5:] = np.arange(5)
    data, ts = _remove_empty_elements(data, ts)
    assert data.size == ts.size
    assert ts.size == 4

    data = np.ones(20).reshape(2, -1)
    ts = np.zeros(10)
    ts[5:] = np.arange(5)
    data, ts = _remove_empty_elements(data, ts)
    assert data.shape[1] == ts.size
    assert ts.size == 4


@pytest.fixture()
def raw_with_annotations_and_first_samp() -> BaseRaw:
    """Raw with annotations and first_samp set."""
    fname = testing.data_path() / "mne-sample" / "sample_audvis_raw.fif"
    raw = read_raw_fif(fname, preload=True)
    events = find_events(raw, stim_channel="STI 014")
    events = events[np.isin(events[:, 2], (1, 2))]  # keep only events with ID 1 and 2
    annotations = annotations_from_events(
        events,
        raw.info["sfreq"],
        event_desc={1: "ignore", 2: "event"},
        first_samp=raw.first_samp,
    )
    annotations.duration += 0.1  # set duration, annotations_from_events sets it to 0
    raw.set_annotations(annotations)
    return raw


@pytest.fixture()
def _mock_lsl_stream_with_annotations_and_first_samp(
    raw_with_annotations_and_first_samp, request, chunk_size
):
    """Create a mock LSL stream streaming events with annotations and first_samp."""
    manager = mp.Manager()
    status = manager.Value("i", 0)
    name = f"P_{request.node.name}"
    process = mp.Process(
        target=_player_mock_lsl_stream,
        args=(raw_with_annotations_and_first_samp, name, chunk_size, status),
    )
    process.start()
    while status.value != 1:
        pass
    yield
    status.value = 0
    process.join(timeout=2)
    process.kill()


@pytest.mark.slow()
@pytest.mark.timeout(30)  # takes under 9s locally
@pytest.mark.usefixtures("_mock_lsl_stream_with_annotations_and_first_samp")
def test_epochs_with_irregular_numerical_event_stream_and_first_samp():
    """Test creating epochs from an event stream from raw with first_samp."""
    event_stream = StreamLSL(10, stype="annotations").connect(acquisition_delay=0.1)
    name = event_stream.name.removesuffix("-annotations")
    stream = StreamLSL(2, name=name).connect(acquisition_delay=0.1)
    stream.info["bads"] = ["MEG 2443"]  # remove bad channel
    epochs = EpochsStream(
        stream,
        bufsize=20,  # number of epoch held in the buffer
        event_id=None,
        event_channels="event",  # this argument now selects the events of interest
        event_stream=event_stream,
        tmin=-0.2,
        tmax=0.5,
        baseline=(None, 0),
        picks="grad",
    ).connect(acquisition_delay=0.1)
    while epochs.n_new_epochs == 0:
        time.sleep(0.2)
    n = epochs.n_new_epochs
    while epochs.n_new_epochs == n:
        time.sleep(0.2)
    n2 = epochs.n_new_epochs
    assert n < n2
    while epochs.n_new_epochs == n2:
        time.sleep(0.2)
    n3 = epochs.n_new_epochs
    assert n2 < n3
    epochs.get_data()
    assert epochs.n_new_epochs == 0
    epochs.disconnect()
    stream.disconnect()
    event_stream.disconnect()
