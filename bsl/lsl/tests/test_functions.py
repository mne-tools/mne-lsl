import uuid

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

    sinfo = StreamInfo("test", "", 1, 0.0, "int8", uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(
            sinfo,
        )
        streams = resolve_streams(timeout=5)
        assert isinstance(streams, list)
        assert len(streams) == 1
    except Exception as error:
        raise error
    finally:
        del outlet
