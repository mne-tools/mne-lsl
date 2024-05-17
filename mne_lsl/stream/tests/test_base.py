import pytest

from .._base import _check_baseline, _ensure_event_id_dict


def test_ensure_event_id_dict():
    """Test validation of event dictionary."""
    assert _ensure_event_id_dict(5) == {"5": 5}
    assert _ensure_event_id_dict({"5": 5}) == {"5": 5}
    assert _ensure_event_id_dict("test") == {"test": "test"}
    assert _ensure_event_id_dict({"test": "test"}) == {"test": "test"}
    assert _ensure_event_id_dict({"t": 5, "t2": "101"}) == {"t": 5, "t2": "101"}

    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        _ensure_event_id_dict("")
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        _ensure_event_id_dict(0)
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        _ensure_event_id_dict(-101)
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        _ensure_event_id_dict({"5": 0})
    with pytest.raises(ValueError, match="'event_id' must be a non-empty string"):
        _ensure_event_id_dict({"101": ""})

    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict(5.5)
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict(None)
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict({"5": 5.5})
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict({"101": None})
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_event_id_dict({5: "test"})


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
