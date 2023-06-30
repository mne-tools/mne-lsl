import uuid

import pytest

from bsl.lsl import (
    StreamInfo,
    StreamOutlet,
    library_version,
    local_clock,
    protocol_version,
    resolve_streams,
)
from bsl.lsl.load_liblsl import _VERSION_MAX, _VERSION_MIN, _VERSION_PROTOCOL


def test_library_version():
    """Test retrieval of library version."""
    version = library_version()
    assert isinstance(version, int)
    assert _VERSION_MIN <= version <= _VERSION_MAX


def test_protocol_version():
    """Test retrieval of protocol version."""
    version = protocol_version()
    assert isinstance(version, int)
    assert version == _VERSION_PROTOCOL


def test_local_clock():
    """Test retrieval of local (client) LSL clock."""
    ts = local_clock()
    assert isinstance(ts, float)
    assert ts >= 0
    assert local_clock() >= ts


def test_resolve_streams():
    """Test detection of streams on the network."""
    streams = resolve_streams()
    assert isinstance(streams, list)
    assert len(streams) == 0

    # detect all streams
    sinfo = StreamInfo("test", "", 1, 0.0, "int8", uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo)
        streams = resolve_streams(timeout=2)
        assert isinstance(streams, list)
        assert len(streams) == 1
        assert streams[0] == sinfo
    except Exception as error:
        raise error
    finally:
        try:
            del outlet
        except Exception:
            pass

    # detect streams by properties
    sinfo1 = StreamInfo("test1", "", 1, 0.0, "int8", "")
    sinfo2 = StreamInfo("test1", "Markers", 1, 0, "int8", "")
    sinfo3 = StreamInfo("test2", "", 1, 0.0, "int8", "")

    try:
        outlet1 = StreamOutlet(sinfo1)
        outlet2 = StreamOutlet(sinfo2)
        outlet3 = StreamOutlet(sinfo3)

        streams = resolve_streams(timeout=2)
        assert len(streams) == 3
        assert sinfo1 in streams
        assert sinfo2 in streams
        assert sinfo3 in streams

        streams = resolve_streams(name="test1", minimum=2)
        assert len(streams) == 2
        assert sinfo1 in streams
        assert sinfo2 in streams

        streams = resolve_streams(name="test1", minimum=1)
        assert len(streams) == 1
        assert sinfo1 in streams or sinfo2 in streams

        streams = resolve_streams(stype="Markers")
        assert len(streams) == 1
        assert sinfo2 in streams

        streams = resolve_streams(name="test2", minimum=2, timeout=1)
        assert len(streams) == 1
        assert sinfo3 in streams

        streams = resolve_streams(name="test1", stype="Markers")
        assert len(streams) == 1
        assert sinfo2 in streams
    except Exception as error:
        raise error
    finally:
        del outlet1
        del outlet2
        del outlet3

    with pytest.raises(
        ValueError, match="'timeout' must be a strictly positive integer"
    ):
        resolve_streams(timeout=-1)
    with pytest.raises(
        ValueError, match="'minimum' must be a strictly positive integer"
    ):
        resolve_streams(name="test", minimum=-1)
