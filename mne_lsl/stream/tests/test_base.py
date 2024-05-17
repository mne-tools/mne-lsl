import pytest
from mne import create_info

from .._base import (
    check_baseline,
    check_reject_flat,
    check_reject_tmin_tmax,
    ensure_event_id_dict,
)


def test_ensure_event_id_dict():
    """Test validation of event dictionary."""
    assert ensure_event_id_dict(5) == {"5": 5}
    assert ensure_event_id_dict({"5": 5}) == {"5": 5}
    assert ensure_event_id_dict("test") == {"test": "test"}
    assert ensure_event_id_dict({"test": "test"}) == {"test": "test"}
    assert ensure_event_id_dict({"t": 5, "t2": "101"}) == {"t": 5, "t2": "101"}

    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        ensure_event_id_dict("")
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        ensure_event_id_dict(0)
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        ensure_event_id_dict(-101)
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        ensure_event_id_dict({"5": 0})
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        ensure_event_id_dict({"101": ""})

    with pytest.raises(TypeError, match="must be an instance of"):
        ensure_event_id_dict(5.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        ensure_event_id_dict(None)
    with pytest.raises(TypeError, match="must be an instance of"):
        ensure_event_id_dict({"5": 5.5})
    with pytest.raises(TypeError, match="must be an instance of"):
        ensure_event_id_dict({"101": None})
    with pytest.raises(TypeError, match="must be an instance of"):
        ensure_event_id_dict({5: "test"})


def test_check_baseline():
    """Test validation of the baseline."""
    check_baseline(None, -0.2, 0.5)
    check_baseline((None, None), -0.2, 0.5)
    check_baseline((None, 0), -0.2, 0.5)
    check_baseline((0, None), -0.2, 0.5)
    check_baseline((0, 0), -0.2, 0.5)

    with pytest.raises(ValueError, match="The beginning of the baseline period must"):
        check_baseline((-0.2, 0), -0.1, 0.5)
    with pytest.raises(ValueError, match="The end of the baseline period must"):
        check_baseline((-0.2, 0.8), -0.2, 0.6)

    with pytest.raises(TypeError, match="must be an instance of"):
        check_baseline((-0.2, "test"), -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        check_baseline(("test", 0.5), -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        check_baseline(("test", "test"), -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        check_baseline(101, -0.2, 0.5)


@pytest.fixture(scope="session")
def info():
    """A simple info object with 10 EEG channels."""
    return create_info(
        ch_names=[f"ch{i}" for i in range(10)], sfreq=100, ch_types="eeg"
    )


def test_check_reject_flat(info):
    """Test validation of the rejection dictionaries."""
    check_reject_flat(None, None, info)
    check_reject_flat(None, {"eeg": 1e-6}, info)
    check_reject_flat({"eeg": 1e-6}, None, info)
    check_reject_flat({"eeg": 1e-6}, {"eeg": 1e-6}, info)

    with pytest.raises(ValueError, match="peak-to-peak rejection value"):
        check_reject_flat({"eeg": -1e-6}, None, info)
    with pytest.raises(ValueError, match="flat rejection value"):
        check_reject_flat(None, {"eeg": -1e-6}, info)
    with pytest.raises(ValueError, match="channel type '.*' in the .* is not part"):
        check_reject_flat(None, {"eog": 1e-6}, info)
    with pytest.raises(ValueError, match="channel type '.*' in the .* is not part"):
        check_reject_flat({"eog": 1e-6}, None, info)

    with pytest.raises(TypeError, match="must be an instance of"):
        check_reject_flat(101, None, info)
    with pytest.raises(TypeError, match="must be an instance of"):
        check_reject_flat(None, 101, info)
    with pytest.raises(TypeError, match="must be an instance of"):
        check_reject_flat({"eeg": "test"}, None, info)
    with pytest.raises(TypeError, match="must be an instance of"):
        check_reject_flat(None, {"eeg": "test"}, info)


def test_check_reject_tmin_tmax():
    """Test validatin of rejection time windows."""
    check_reject_tmin_tmax(None, None, -0.2, 0.5)
    check_reject_tmin_tmax(-0.2, 0.5, -0.2, 0.5)
    check_reject_tmin_tmax(None, 0.5, -0.2, 0.5)
    check_reject_tmin_tmax(-0.2, None, -0.2, 0.5)
    check_reject_tmin_tmax(0, 0.1, -0.2, 0.5)

    with pytest.raises(ValueError, match="The beginning of the rejection time window"):
        check_reject_tmin_tmax(-0.5, 0.5, -0.2, 0.5)
    with pytest.raises(ValueError, match="end of the epoch period"):
        check_reject_tmin_tmax(-0.2, 0.8, -0.2, 0.5)
    with pytest.raises(ValueError, match="must be greater than the beginning"):
        check_reject_tmin_tmax(0.5, -0.2, -0.2, 0.5)

    with pytest.raises(TypeError, match="must be an instance of"):
        check_reject_tmin_tmax("test", None, -0.2, 0.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        check_reject_tmin_tmax(None, "test", -0.2, 0.5)
