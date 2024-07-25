from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import create_info
from numpy.testing import assert_allclose

from ..epochs import (
    _check_baseline,
    _check_reject_flat,
    _check_reject_tmin_tmax,
    _ensure_detrend_int,
    _ensure_event_id_dict,
    _prune_events,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_ensure_event_id_dict():
    """Test validation of event dictionary."""
    assert _ensure_event_id_dict(5) == {"5": 5}
    assert _ensure_event_id_dict({"5": 5}) == {"5": 5}

    with pytest.raises(ValueError, match="must be a positive integer or a dictionary"):
        _ensure_event_id_dict(0)
    with pytest.raises(ValueError, match="must be a positive integer or a dictionary"):
        _ensure_event_id_dict(-101)
    with pytest.raises(ValueError, match="must be a positive integer or a dictionary"):
        _ensure_event_id_dict({"5": 0})

    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict(5.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict(None)
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict({"5": 5.5})
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict({"101": None})


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


def test_ensure_detrend_int():
    """Test validation of detrend."""
    assert _ensure_detrend_int(None) is None
    assert _ensure_detrend_int(0) == 0
    assert _ensure_detrend_int(1) == 1
    assert _ensure_detrend_int("constant") == 0
    assert _ensure_detrend_int("linear") == 1

    with pytest.raises(ValueError, match="detrend argument must be"):
        _ensure_detrend_int("test")

    with pytest.raises(TypeError, match="must be an integer"):
        _ensure_detrend_int(5.5)


def test_find_events_in_stim_channels():
    """Test finding events in stimulation channels."""


@pytest.fixture()
def n_events() -> int:
    """Return the number of events."""
    return 10


@pytest.fixture()
def events(n_events: int) -> NDArray[np.int64]:
    """Return a simple event array.

    An event is present every 10 samples, cycling between the values (1, 2, 3).
    """
    events = np.zeros((n_events, 3), dtype=np.int64)
    for k in range(events.shape[0]):
        events[k, :] = [10 * (k + 1), 0, k % 3 + 1]
    return events


def test_prune_events(events: NDArray[np.int64]):
    """Test pruning events."""
    ts = np.arange(10000, 11000, 1.8)
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, None, None)
    assert_allclose(events_, events)
    # test pruning events outside of the event_id dictionary
    events_ = _prune_events(events, dict(a=1, c=3), 10, ts, None, None)
    assert sorted(np.unique(events_[:, 2])) == [1, 3]
    # test pruning events that can't fit in the buffer
    ts = np.arange(5)
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, None, None)
    assert events_.size == 0
    ts = np.arange(10000, 11000, 1.8)  # ts.size == 556
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 500, ts, None, None)
    assert events_[-1, 0] + 500 <= ts.size
    assert events_[-1, 0] == 50  # events @ 60, 70, 80, ... should be dropped
    # test pruning events that have already been moved to the buffer
    ts = np.arange(10000, 11000, 1.8)  # ts.size == 556
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, ts[events[3, 0]], None)
    assert_allclose(events_, events[4:, :])
    # test pruning events from an event stream, which converts the index to index in ts
    ts = np.arange(1000)
    ts_events = np.arange(500) * 2 + 0.5  # mock a different sampling frequency
    events_ = _prune_events(events, dict(a=1, b=2, c=3), 10, ts, None, ts_events)
    assert_allclose(events_[:, 2], events[:, 2])
    # with the half sampling rate + 0.5 set above, we should be selecting:
    # from: 10, 20, 30, 40, ... corresponding to 20.5, 40.5, 60.5, ...
    # to: 21, 41, 61, ... corresponding to 20, 40, 60, ...
    assert_allclose(events_[:, 0], np.arange(20, 20 * (events_[:, 0].size + 1), 20) + 1)
