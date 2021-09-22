from bsl.stream_receiver._buffer import Buffer


def test_buffer():
    """Test Buffer class used by the StreamReceiver."""
    # buffer creation
    b = Buffer(bufsize=100, winsize=100)
    assert b.bufsize == 100
    assert b.winsize == 100
    assert len(b.data) == 0
    assert len(b.timestamps) == 0

    # add elements less than buffer limit
    chs = 3
    samples = 5
    fake_data = [[1] * chs] * samples
    fake_timestamps = list(range(samples))
    b.fill(fake_data, fake_timestamps)
    assert len(b.data) == samples
    assert len(b.timestamps) == samples
    assert all(elt == [1]*chs for elt in b.data)
    assert b.timestamps == fake_timestamps

    # add elements above buffer limit
    samples = 97
    fake_data = [[2] * chs] * samples
    fake_timestamps = list(range(samples))
    b.fill(fake_data, fake_timestamps)
    assert len(b.data) == b.bufsize
    assert len(b.timestamps) == b.bufsize
    assert all(elt == [1]*chs for elt in b.data[:3])
    assert all(elt == [2]*chs for elt in b.data[3:])

    # reset buffer
    b.reset_buffer()
    assert len(b.data) == 0
    assert len(b.timestamps) == 0
