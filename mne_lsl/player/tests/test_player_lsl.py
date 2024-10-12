from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import annotations_from_events, create_info, find_events
from mne.io import RawArray, read_raw_fif
from mne.utils import check_version
from numpy.testing import assert_allclose

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF
else:
    from mne.io.constants import FIFF

from mne_lsl.lsl import StreamInlet, local_clock, resolve_streams
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.utils._tests import match_stream_and_raw_data

if TYPE_CHECKING:
    from mne.io import BaseRaw


def test_player(fname, raw, close_io, chunk_size, request):
    """Test a working and valid player."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    player = Player(fname, chunk_size=chunk_size, name=name, source_id=source_id)
    assert "OFF" in player.__repr__()
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]
    player.start()
    assert "ON" in player.__repr__()
    streams = resolve_streams(timeout=2)
    assert (name, source_id) in [(stream.name, stream.source_id) for stream in streams]

    # try double start
    with pytest.warns(RuntimeWarning, match="player is already started"):
        player.start()

    # connect an inlet to the player
    inlet = _create_inlet(name, source_id)
    data, ts = inlet.pull_chunk()
    now = local_clock()
    # between the chunk size and the slow CIs, we can't expect precise timings
    assert_allclose(now, ts[-1], rtol=1e-3, atol=1)
    assert data.shape[1] == len(player.info["ch_names"])

    # check sampling rate
    fs = 1 / np.diff(ts)
    assert_allclose(fs, player.info["sfreq"])

    # check inlet information
    sinfo = inlet.get_sinfo()
    assert sinfo.n_channels == len(player.info["ch_names"])
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_types() == player.get_channel_types()
    assert sinfo.get_channel_units() == [
        str(int(ch["unit_mul"])) for ch in player.info["chs"]
    ]

    # check player vs raw
    assert player.info["sfreq"] == raw.info["sfreq"]
    assert player.info["ch_names"] == raw.info["ch_names"]
    assert player.get_channel_types() == raw.get_channel_types()
    assert [ch["unit_mul"] for ch in player.info["chs"]] == [
        ch["unit_mul"] for ch in raw.info["chs"]
    ]

    # check that the returned data array is in raw
    match_stream_and_raw_data(data.T, raw)
    close_io()
    player.stop()


def test_player_context_manager(fname, chunk_size, request):
    """Test a working and valid player as context manager."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]
    with Player(fname, chunk_size=chunk_size, name=name, source_id=source_id):
        streams = resolve_streams(timeout=2)
        assert (name, source_id) in [
            (stream.name, stream.source_id) for stream in streams
        ]
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]


def test_player_context_manager_raw(raw, chunk_size, request):
    """Test a working and valid player as context manager from a raw object."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]
    with Player(raw, chunk_size=chunk_size, name=name, source_id=source_id) as player:
        streams = resolve_streams(timeout=2)
        assert (name, source_id) in [
            (stream.name, stream.source_id) for stream in streams
        ]
        assert player.info["ch_names"] == raw.info["ch_names"]
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]

    with pytest.warns(RuntimeWarning, match="raw file has no annotations"):
        with Player(
            raw, chunk_size=chunk_size, name=name, source_id=source_id, annotations=True
        ) as player:
            streams = resolve_streams(timeout=2)
            assert (name, source_id) in [
                (stream.name, stream.source_id) for stream in streams
            ]
            assert player.info["ch_names"] == raw.info["ch_names"]


def test_player_context_manager_raw_annotations(raw_annotations, chunk_size, request):
    """Test a working player as context manager from a raw object with annotations."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]
    with Player(
        raw_annotations,
        chunk_size=chunk_size,
        name=name,
        source_id=source_id,
        annotations=False,
    ) as player:
        assert player.running
        streams = resolve_streams(timeout=2)
        assert (name, source_id) in [
            (stream.name, stream.source_id) for stream in streams
        ]
        assert player.info["ch_names"] == raw_annotations.info["ch_names"]
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]

    with Player(
        raw_annotations, chunk_size=chunk_size, name=name, source_id=source_id
    ) as player:
        assert player.running
        streams = resolve_streams(timeout=2)
        assert (name, source_id) in [
            (stream.name, stream.source_id) for stream in streams
        ]
        assert (f"{name}-annotations", source_id) in [
            (stream.name, stream.source_id) for stream in streams
        ]
        assert player.info["ch_names"] == raw_annotations.info["ch_names"]
    streams = resolve_streams(timeout=0.1)
    assert name not in [stream.name for stream in streams]


def test_player_invalid_arguments(fname):
    """Test creation of a player with invalid arguments."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Player("invalid-fname.something")
    with pytest.raises(ValueError, match="Unsupported file type"):
        Player(Path(__file__))
    with pytest.raises(TypeError, match="'name' must be an instance of str or None"):
        Player(fname, name=101)
    with pytest.raises(TypeError, match="'chunk_size' must be an integer"):
        Player(fname, name="101", chunk_size=101.0)
    with pytest.raises(ValueError, match="strictly positive integer"):
        Player(fname, name="101", chunk_size=-101)


def test_player_stop_invalid(fname, chunk_size, request):
    """Test stopping a player that is not started."""
    player = Player(
        fname, chunk_size=chunk_size, name=f"P_{request.node.name}_{uuid.uuid4().hex}"
    )
    with pytest.raises(RuntimeError, match="The player is not started"):
        player.stop()
    player.start()
    player.stop()


def _create_inlet(name: str, source_id: str) -> StreamInlet:
    """Create an inlet to the open-stream."""
    streams = resolve_streams(timeout=2)
    streams = [
        stream
        for stream in streams
        if stream.name == name and stream.source_id == source_id
    ]
    assert len(streams) == 1
    inlet = StreamInlet(streams[0])
    inlet.open_stream(timeout=10)
    return inlet


@pytest.fixture()
def mock_lsl_stream(fname: Path, chunk_size, request):
    """Create a mock LSL stream for testing."""
    # nest the PlayerLSL import to first write the temporary LSL configuration file
    from mne_lsl.player import PlayerLSL  # noqa: E402

    with PlayerLSL(
        fname,
        chunk_size=chunk_size,
        name=f"P_{request.node.name}",
        source_id=uuid.uuid4().hex,
    ) as player:
        yield player


@pytest.mark.slow()
def test_player_unit(mock_lsl_stream, raw, close_io):
    """Test getting and setting the player channel units."""
    player = mock_lsl_stream
    assert player.get_channel_types() == raw.get_channel_types()
    assert player.get_channel_types(unique=True) == raw.get_channel_types(unique=True)
    ch_units = player.get_channel_units()
    assert ch_units == [(FIFF.FIFF_UNIT_NONE, FIFF.FIFF_UNITM_NONE)] + [
        (FIFF.FIFF_UNIT_V, FIFF.FIFF_UNITM_NONE)
    ] * (len(player.ch_names) - 1)

    # try setting channel units on a started player
    with pytest.raises(RuntimeError, match="player is already started"):
        player.set_channel_units({"F7": -6, "Fpz": "uv", "Fp2": "microvolts"})
    inlet = _create_inlet(player.name, player.source_id)
    data, _ = inlet.pull_chunk()
    match_stream_and_raw_data(data.T, raw)
    close_io()
    player.stop()

    # try setting channel units after stopping the player
    player.set_channel_units({"F7": -6, "Fpz": "uv", "Fp2": "microvolts"})
    player.start()
    inlet = _create_inlet(player.name, player.source_id)
    data, _ = inlet.pull_chunk()
    raw_ = raw.copy().apply_function(lambda x: x * 1e6, picks=["F7", "Fpz", "Fp2"])
    match_stream_and_raw_data(data.T, raw_)
    close_io()
    player.stop()

    # try re-setting the channel unit
    player.set_channel_units({"F7": -3})
    player.start()
    inlet = _create_inlet(player.name, player.source_id)
    data, _ = inlet.pull_chunk()
    raw_ = raw.copy()
    raw_.apply_function(lambda x: x * 1e3, picks="F7")
    raw_.apply_function(lambda x: x * 1e6, picks=["Fpz", "Fp2"])
    match_stream_and_raw_data(data.T, raw_)
    close_io()
    player.stop()


@pytest.mark.slow()
def test_player_rename_channels(mock_lsl_stream, raw, close_io):
    """Test channel renaming."""
    player = mock_lsl_stream
    assert player._sinfo.get_channel_names() == player.info["ch_names"]
    with pytest.raises(RuntimeError, match="player is already started"):
        player.rename_channels(mapping={"F7": "EEG1"})
    inlet = _create_inlet(player.name, player.source_id)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    close_io()
    player.stop()

    # test changing channel names
    player.rename_channels({"F7": "EEG1", "Fp2": "EEG2"})
    raw_ = raw.copy().rename_channels({"F7": "EEG1", "Fp2": "EEG2"})
    player.start()
    inlet = _create_inlet(player.name, player.source_id)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    close_io()
    player.stop()

    # test re-changing the channel names
    player.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    raw_.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    player.start()
    inlet = _create_inlet(player.name, player.source_id)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    close_io()
    player.stop()


@pytest.mark.slow()
def test_player_set_channel_types(mock_lsl_stream, raw, close_io):
    """Test channel type setting."""
    player = mock_lsl_stream
    assert player._sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert player.get_channel_types(unique=False) == raw.get_channel_types(unique=False)
    with pytest.raises(RuntimeError, match="player is already started"):
        player.set_channel_types(mapping={"F7": "misc"})
    inlet = _create_inlet(player.name, player.source_id)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    close_io()
    player.stop()

    # test changing types
    player.set_channel_types(mapping={"F7": "eog", "Fp2": "eog"})
    raw_ = raw.copy().set_channel_types(mapping={"F7": "eog", "Fp2": "eog"})
    player.start()
    inlet = _create_inlet(player.name, player.source_id)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert sinfo.get_channel_types() == raw_.get_channel_types(unique=False)
    close_io()
    player.stop()

    # test rechanging types
    player.set_channel_types(mapping={"F7": "eeg", "Fp2": "ecg"})
    raw_ = raw.copy().set_channel_types(mapping={"Fp2": "ecg"})
    player.start()
    inlet = _create_inlet(player.name, player.source_id)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert sinfo.get_channel_types() == raw_.get_channel_types(unique=False)
    close_io()
    player.stop()

    # test unique
    assert sorted(player.get_channel_types(unique=True)) == sorted(
        raw.get_channel_types(unique=True)
    )


def test_player_anonymize(fname, chunk_size, request):
    """Test anonymization."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    player = Player(fname, chunk_size=chunk_size, name=name, source_id=source_id)
    assert player.name == name
    assert player.source_id == source_id
    assert player.fname == fname
    player.info["subject_info"] = dict(id=101, first_name="Mathieu", sex=1)
    with pytest.warns(RuntimeWarning, match="partially implemented"):
        player.anonymize()
    assert player.info["subject_info"] == dict(id=0, first_name="mne_anonymize", sex=0)
    player.start()
    assert "ON" in player.__repr__()
    with pytest.raises(RuntimeError, match="player is already started"):
        player.anonymize()
    player.stop()


def test_player_set_meas_date(fname, chunk_size, request):
    """Test player measurement date."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    player = Player(fname, chunk_size=chunk_size, name=name, source_id=source_id)
    assert player.name == name
    assert player.source_id == source_id
    assert player.fname == fname
    assert player.info["meas_date"] is None
    with pytest.warns(RuntimeWarning, match="partially implemented"):
        meas_date = datetime(2023, 1, 25, tzinfo=timezone.utc)
        player.set_meas_date(meas_date)
    assert player.info["meas_date"] == meas_date
    player.start()
    assert "ON" in player.__repr__()
    with pytest.raises(RuntimeError, match="player is already started"):
        player.set_meas_date(datetime(2020, 1, 25, tzinfo=timezone.utc))
    assert player.info["meas_date"] == meas_date
    player.stop()


@pytest.mark.slow()
def test_player_annotations(raw_annotations, close_io, chunk_size, request):
    """Test player with annotations."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    annotations = sorted(set(raw_annotations.annotations.description))
    player = Player(
        raw_annotations, chunk_size=chunk_size, name=name, source_id=source_id
    )
    assert f"Player: {name}" in repr(player)
    assert player.name == name
    assert player.source_id == source_id
    assert player.fname == Path(raw_annotations.filenames[0])
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]
    player.start()
    streams = resolve_streams(timeout=2)
    assert (name, source_id) in [(stream.name, stream.source_id) for stream in streams]
    assert (f"{name}-annotations", source_id) in [
        (stream.name, stream.source_id) for stream in streams
    ]

    # find annotation stream and open an inlet
    inlet_annotations = _create_inlet(f"{name}-annotations", source_id)

    # compare inlet stream info and annotations
    sinfo = inlet_annotations.get_sinfo()
    assert sinfo.n_channels == len(annotations)
    assert sinfo.get_channel_names() == annotations
    assert sinfo.get_channel_types() == ["annotations"] * sinfo.n_channels
    assert sinfo.get_channel_units() == ["none"] * sinfo.n_channels

    # compare content
    data, ts = inlet_annotations.pull_chunk(timeout=1)
    assert data.size != 0
    assert ts.size == data.shape[0]
    # compare with a Stream object for simplicity
    stream = Stream(bufsize=40, stype="annotations", source_id=source_id)
    stream.connect(processing_flags=["clocksync"])
    assert stream.info["ch_names"] == annotations
    assert stream.get_channel_types() == ["misc"] * sinfo.n_channels
    time.sleep(3)  # acquire some annotations
    for single, duration in zip(
        ("bad_test", "test2", "test3"), (0.4, 0.1, 0.05), strict=False
    ):
        data, ts = stream.get_data(picks=single)
        data = data.squeeze()
        assert ts.size == data.size
        idx = np.where(data != 0.0)[0]
        assert_allclose(data[idx], [duration] * idx.size)
        assert_allclose(np.diff(ts[idx]), 2, atol=1e-2)
    time.sleep(3)
    data, ts = stream.get_data(picks="test1")
    data = data.squeeze()
    idx = np.where(data != 0.0)[0]
    assert_allclose(np.unique(data[idx]), [0.2, 0.55])
    idx = np.where(data == 0.2)[0]
    diff = np.diff(ts[idx])
    expected = np.array([1.6, 0.3, 0.1])
    start = np.where(1 <= diff)[0][0]
    end = diff.size if diff.size <= start + 3 else start + 3
    assert_allclose(diff[start:end], expected[: end - start], atol=1e-2)

    # clean-up
    stream.disconnect()
    close_io()
    player.stop()


@pytest.fixture()
def raw_annotations_1000_samples() -> BaseRaw:
    """Return a 1000 sample raw object with annotations."""
    n_samples = 1000
    data = np.zeros((2, n_samples), dtype=np.float32)
    data[0, :] = np.arange(n_samples)  # index of the sample within the raw object
    for pos in (100, 500, 700):
        data[-1, pos : pos + 10] = 1  # trigger channel at the end
    info = create_info(["ch0", "trg"], 1000, ["eeg", "stim"])
    raw = RawArray(data, info)
    events = find_events(raw, "trg")
    annotations = annotations_from_events(
        events, raw.info["sfreq"], event_desc={1: "event"}, first_samp=raw.first_samp
    )
    annotations.duration += 0.01
    return raw.drop_channels("trg").set_annotations(annotations)


@pytest.mark.slow()
def test_player_annotations_multiple_of_chunk_size(
    raw_annotations_1000_samples, chunk_size, request
):
    """Test player with annotations, chunk-size is a multiple of the raw size."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    raw = raw_annotations_1000_samples
    assert raw.times.size % chunk_size == 0
    player = Player(raw, chunk_size=chunk_size, name=name, source_id=source_id)
    player.start()
    time.sleep((raw.times.size / raw.info["sfreq"]) * 1.8)
    streams = resolve_streams(timeout=2)
    assert (name, source_id) in [(stream.name, stream.source_id) for stream in streams]
    assert (f"{name}-annotations", source_id) in [
        (stream.name, stream.source_id) for stream in streams
    ]
    time.sleep((raw.times.size / raw.info["sfreq"]) * 1.8)
    streams = resolve_streams(timeout=2)
    assert (name, source_id) in [(stream.name, stream.source_id) for stream in streams]
    assert (f"{name}-annotations", source_id) in [
        (stream.name, stream.source_id) for stream in streams
    ]
    player.stop()


@pytest.mark.slow()
def test_player_n_repeat(raw, chunk_size, request):
    """Test argument 'n_repeat'."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    player = Player(
        raw, chunk_size=chunk_size, n_repeat=1, name=name, source_id=source_id
    )
    player.start()
    time.sleep((raw.times.size / raw.info["sfreq"]) * 1.8)
    assert player._executor is None
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [
        (stream.name, stream.source_id) for stream in streams
    ]
    with pytest.raises(RuntimeError, match="player is not started."):
        player.stop()
    source_id2 = uuid.uuid4().hex
    assert source_id != source_id2
    player = Player(
        raw, chunk_size=chunk_size, n_repeat=4, name=name, source_id=source_id2
    )
    assert player.n_repeat == 4
    player.start()
    assert player.n_repeat == 4
    assert player.running
    time.sleep((raw.times.size / raw.info["sfreq"]) * 1.8)
    assert player._executor is not None
    streams = resolve_streams(timeout=2)
    assert (name, source_id2) in [(stream.name, stream.source_id) for stream in streams]
    player.stop()


@pytest.mark.slow()
def test_player_n_repeat_mmapped(fname, close_io, chunk_size, request):
    """Test argument 'n_repeat' with non-preloaded raw."""
    raw = read_raw_fif(fname, preload=False).crop(0, 1)  # crop from 2s to 1s
    index = raw.get_data(picks=0).squeeze().astype(int)
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    n_iterations = 2  # test for 2 repeats
    timeout = (raw.times.size / raw.info["sfreq"]) * (n_iterations + 1)
    # set an arbitrary 'n_repeat' large value to test the argument on a short file.
    with Player(raw, chunk_size, n_repeat=20, name=name, source_id=source_id) as player:
        streams = resolve_streams(timeout=2)
        assert (name, source_id) in [
            (stream.name, stream.source_id) for stream in streams
        ]
        inlet = _create_inlet(name, source_id)
        start_idx = player._start_idx
        # the variable '_n_repeated' should start at 1, except if inlet creation was too
        # slow which could happen when playing short files.
        start_repeat = player._n_repeated
        current_repeat = start_repeat
        start_time = time.time()
        # we want to play the file 2 times, i.e. '_n_repeated' will be set to 1 and 2,
        # thus the loop exit when '_n_repeated' equals 3 (if start_repeat == 1).
        while player._n_repeated < start_repeat + n_iterations:
            data, _ = inlet.pull_chunk()
            if player._start_idx < start_idx:  # are we looping?
                current_repeat += 1
                last_sample_idx = np.where(data[:, 0] == index[-1])[0]
                assert last_sample_idx.size == 1  # sanity-check
                # check indexes before repeat
                index_data = data[: last_sample_idx[0] + 1, 0]
                assert_allclose(
                    index_data, np.arange(index[-1] - index_data.size, index[-1]) + 1
                )
                # check indexes after repeat
                index_data = data[last_sample_idx[0] + 1 :, 0]
                assert_allclose(index_data, np.arange(index_data.size))
            start_idx = player._start_idx
            assert player._n_repeated == current_repeat  # test incrementation
            if timeout < time.time() - start_time:
                raise RuntimeError("Timeout reached.")
            time.sleep(0.1)
        close_io()


def test_player_n_repeat_invalid(raw):
    """Test invalid argument 'n_repeat'."""
    with pytest.raises(ValueError, match="strictly positive integer"):
        Player(raw, n_repeat=0)
    with pytest.raises(ValueError, match="strictly positive integer"):
        Player(raw, n_repeat=-1)
    with pytest.raises(TypeError, match="must be an integer"):
        Player(raw, n_repeat="1")


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS", "") == "true", reason="Unreliable on CIs."
)
def test_player_push_sample(fname, request):
    """Test pushing individual sample with chunk_size=1."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [(stream.name, source_id) for stream in streams]
    with Player(fname, chunk_size=1, name=name):
        streams = resolve_streams(timeout=0.5)
        assert (name, source_id) in [(stream.name, source_id) for stream in streams]
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [(stream.name, source_id) for stream in streams]


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS", "") == "true", reason="Unreliable on CIs."
)
def test_player_push_last_sample(fname, caplog, request):
    """Test pushing the last sample."""
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    player = Player(fname, chunk_size=1, n_repeat=1, name=name, source_id=source_id)
    caplog.clear()
    player.start()
    streams = resolve_streams(timeout=0.5)
    assert (name, source_id) in [(stream.name, source_id) for stream in streams]
    while player.running:
        time.sleep(0.1)
    # 'IndexError: index 0 is out of bounds' would be raised it the last chunk pushed
    # was empty.
    assert "index 0 is out of bounds" not in caplog.text
    streams = resolve_streams(timeout=0.1)
    assert (name, source_id) not in [(stream.name, source_id) for stream in streams]
