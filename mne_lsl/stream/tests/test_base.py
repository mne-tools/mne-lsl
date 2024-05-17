import pytest

from .._base import _ensure_event_id_dict


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
