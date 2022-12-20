import pickle
import time
from pathlib import Path

import mne
import pytest

from bsl import StreamPlayer, StreamRecorder, logger, set_log_level
from bsl.datasets import eeg_resting_state
from bsl.utils._tests import requires_eeg_resting_state_dataset

set_log_level("INFO")
logger.propagate = True


def _check_recorded_files(record_dir, eve_file, stream, fif_subdir):
    """Use eve_file to retrieve the file name stem and check if the recorded
    files exists."""
    fname_stem = eve_file.stem.split("-eve")[0]
    fname_pcl = record_dir / f"{fname_stem}-{stream}-raw.pcl"
    if fif_subdir:
        fname_fif = record_dir / "fif" / f"{fname_stem}-{stream}-raw.fif"
    else:
        fname_fif = record_dir / f"{fname_stem}-{stream}-raw.fif"
    assert fname_pcl.exists()
    assert fname_fif.exists()


def _check_recorded_files_content(
    record_dir,
    eve_file,
    stream,
    fif_subdir,
    dataset,
    record_duration,
):
    """Checks the recorded files content."""
    # patch because of the delay added in StreamInlet.open_stream()
    # due to the issue https://github.com/sccn/liblsl/issues/176 on the C++ lib
    record_duration += 0.5

    fname_stem = eve_file.stem.split("-eve")[0]
    fname_pcl = record_dir / f"{fname_stem}-{stream}-raw.pcl"
    if fif_subdir:
        fname_fif = record_dir / "fif" / f"{fname_stem}-{stream}-raw.fif"
    else:
        fname_fif = record_dir / f"{fname_stem}-{stream}-raw.fif"
    raw = mne.io.read_raw_fif(fname=dataset.data_path(), preload=True)
    with open(fname_pcl, "rb") as inp:
        raw_pcl = pickle.load(inp)
    raw_fif = mne.io.read_raw_fif(fname_fif, preload=True)
    assert raw.ch_names == raw_pcl["ch_names"] == raw_fif.ch_names
    assert raw.info["sfreq"] == raw_pcl["sample_rate"] == raw_fif.info["sfreq"]
    assert raw_pcl["signals"].shape[::-1] == raw_fif.get_data().shape
    # some delay is introduce by the process operations
    assert 0 <= raw_fif.n_times / raw_fif.info["sfreq"] - record_duration < 0.2


@requires_eeg_resting_state_dataset
def test_stream_recorder(tmp_path, caplog):
    """Test recording capability of the stream recorder."""
    stream = "StreamPlayer"
    record_duration = 0.5  # seconds
    dataset = eeg_resting_state
    fif_subdir = False

    # Test default call
    with StreamPlayer(stream, dataset.data_path()):
        recorder = StreamRecorder(
            record_dir=tmp_path, fif_subdir=fif_subdir, verbose=False
        )
        assert recorder._state.value == 0
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        assert eve_file is not None
        assert recorder._state.value == 1
        time.sleep(record_duration)
        recorder.stop()
        assert "Waiting for StreamRecorder process to finish." in caplog.text
        assert "Recording finished." in caplog.text
        assert recorder._eve_file is None
        assert recorder._process is None
        assert recorder._state.value == 0

    _check_recorded_files(tmp_path, eve_file, stream, fif_subdir)
    _check_recorded_files_content(
        tmp_path, eve_file, stream, fif_subdir, dataset, record_duration
    )

    # Test stop when not started
    caplog.clear()
    recorder = StreamRecorder(record_dir=tmp_path)
    assert recorder._state.value == 0
    recorder.stop()
    assert recorder._state.value == 0
    assert "StreamRecorder was not started. Skipping." in caplog.text

    # Test context manager
    with StreamPlayer(stream, dataset.data_path()):
        with StreamRecorder(
            record_dir=tmp_path,
            fname="test-context-manager",
            fif_subdir=fif_subdir,
        ):
            time.sleep(record_duration)

    eve_file = tmp_path / "test-context-manager-eve.txt"
    _check_recorded_files(tmp_path, eve_file, stream, fif_subdir)
    _check_recorded_files_content(
        tmp_path, eve_file, stream, fif_subdir, dataset, record_duration
    )


@requires_eeg_resting_state_dataset
def test_recording_multiple_streams(tmp_path):
    """Test multi-stream recording capabilities of the stream recorder."""
    record_duration = 0.5  # seconds
    dataset = eeg_resting_state
    fif_subdir = False

    with StreamPlayer("StreamPlayer1", dataset.data_path()), StreamPlayer(
        "StreamPlayer2", dataset.data_path()
    ):
        # Record only StreamPlayer1
        recorder = StreamRecorder(
            record_dir=tmp_path,
            stream_name="StreamPlayer1",
            fif_subdir=fif_subdir,
            verbose=False,
        )
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()

        _check_recorded_files(tmp_path, eve_file, "StreamPlayer1", fif_subdir)

        # Record only StreamPlayer2
        recorder = StreamRecorder(
            record_dir=tmp_path,
            stream_name="StreamPlayer2",
            fif_subdir=fif_subdir,
            verbose=False,
        )
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()

        _check_recorded_files(tmp_path, eve_file, "StreamPlayer2", fif_subdir)

        # Record both - stream_name = None
        recorder = StreamRecorder(
            record_dir=tmp_path,
            stream_name=None,
            fif_subdir=fif_subdir,
            verbose=False,
        )
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()

        _check_recorded_files(tmp_path, eve_file, "StreamPlayer1", fif_subdir)
        _check_recorded_files(tmp_path, eve_file, "StreamPlayer2", fif_subdir)

        # Record both - stream_name = ['StreamPlayer1', 'StreamPlayer2']
        recorder = StreamRecorder(
            record_dir=tmp_path,
            stream_name=["StreamPlayer1", "StreamPlayer2"],
            fif_subdir=fif_subdir,
            verbose=False,
        )
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()

        _check_recorded_files(tmp_path, eve_file, "StreamPlayer1", fif_subdir)
        _check_recorded_files(tmp_path, eve_file, "StreamPlayer2", fif_subdir)


@requires_eeg_resting_state_dataset
def test_arg_fif_subdir(tmp_path):
    """Test argument fif_subdir."""
    record_duration = 0.5  # seconds
    dataset = eeg_resting_state

    with StreamPlayer("StreamPlayer", dataset.data_path()):

        # False
        fif_subdir = False

        recorder = StreamRecorder(
            record_dir=tmp_path, fif_subdir=fif_subdir, verbose=False
        )
        assert recorder._fif_subdir is False
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorded_files(tmp_path, eve_file, "StreamPlayer", fif_subdir)
        _check_recorded_files_content(
            tmp_path,
            eve_file,
            "StreamPlayer",
            fif_subdir,
            dataset,
            record_duration,
        )

        # True
        fif_subdir = True

        recorder = StreamRecorder(
            record_dir=tmp_path, fif_subdir=fif_subdir, verbose=False
        )
        assert recorder._fif_subdir is True
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorded_files(tmp_path, eve_file, "StreamPlayer", fif_subdir)
        _check_recorded_files_content(
            tmp_path,
            eve_file,
            "StreamPlayer",
            fif_subdir,
            dataset,
            record_duration,
        )


@requires_eeg_resting_state_dataset
def test_arg_verbose(tmp_path):
    """Test argument verbose."""
    record_duration = 0.5  # seconds
    dataset = eeg_resting_state
    fif_subdir = False

    with StreamPlayer("StreamPlayer", dataset.data_path()):

        # False
        verbose = False

        recorder = StreamRecorder(
            record_dir=tmp_path, fif_subdir=False, verbose=verbose
        )
        assert recorder._verbose is False
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorded_files(tmp_path, eve_file, "StreamPlayer", fif_subdir)
        _check_recorded_files_content(
            tmp_path,
            eve_file,
            "StreamPlayer",
            fif_subdir,
            dataset,
            record_duration,
        )

        # True
        verbose = True

        recorder = StreamRecorder(
            record_dir=tmp_path, fif_subdir=False, verbose=verbose
        )
        assert recorder._verbose is True
        recorder.start(blocking=True)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorded_files(tmp_path, eve_file, "StreamPlayer", fif_subdir)
        _check_recorded_files_content(
            tmp_path,
            eve_file,
            "StreamPlayer",
            fif_subdir,
            dataset,
            record_duration,
        )


@requires_eeg_resting_state_dataset
def test_properties(tmp_path):
    """Test the StreamRecorder properties."""
    record_duration = 0.5  # seconds
    dataset = eeg_resting_state

    with StreamPlayer("StreamPlayer", dataset.data_path()):
        recorder = StreamRecorder(
            record_dir=tmp_path,
            fname="test",
            stream_name="StreamPlayer",
            fif_subdir=False,
            verbose=False,
        )

        assert recorder.record_dir == tmp_path
        assert recorder.fname == "test"
        assert recorder.stream_name == "StreamPlayer"
        assert recorder.fif_subdir is False
        assert recorder.verbose is False

        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.record_dir = "new path"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.fname = "new fname"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.stream_name = "new stream name"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.fif_subdir = True
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.verbose = True

        recorder.start(blocking=True)

        assert recorder.record_dir == tmp_path
        assert recorder.fname == "test"
        assert recorder.stream_name == "StreamPlayer"
        assert recorder.fif_subdir is False
        assert recorder.verbose is False

        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.record_dir = "new path"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.fname = "new fname"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.stream_name = "new stream name"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.fif_subdir = True
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.verbose = True

        time.sleep(record_duration)
        recorder.stop()

        assert recorder.record_dir == tmp_path
        assert recorder.fname == "test"
        assert recorder.stream_name == "StreamPlayer"
        assert recorder.fif_subdir is False
        assert recorder.verbose is False

        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.record_dir = "new path"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.fname = "new fname"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.stream_name = "new stream name"
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.fif_subdir = True
        with pytest.raises(AttributeError, match="can't set attribute"):
            recorder.verbose = True


def test_checker_arguments(tmp_path):
    """Test the argument error checking."""
    with pytest.raises(TypeError, match="'stream_name' must be an instance"):
        StreamRecorder(record_dir=tmp_path, stream_name=101)
    with pytest.raises(TypeError, match="'fif_subdir' must be an instance"):
        StreamRecorder(record_dir=tmp_path, fif_subdir=1)
    with pytest.raises(TypeError, match="'verbose' must be an instance"):
        StreamRecorder(record_dir=tmp_path, verbose=1)


def test_checker_record_dir(tmp_path):
    """Test the checker for argument record_dir."""
    # Valid
    recorder = StreamRecorder(record_dir=None)
    assert recorder.record_dir == Path.cwd()
    recorder = StreamRecorder(record_dir=tmp_path)
    assert recorder.record_dir == tmp_path

    # Invalid
    with pytest.raises(TypeError, match="'record_dir' must be an instance"):
        StreamRecorder(record_dir=101)


def test_checker_fname(tmp_path):
    """Test the checker for argument fname."""
    # Valid
    recorder = StreamRecorder(record_dir=tmp_path, fname=None)
    assert recorder.fname is None
    recorder = StreamRecorder(record_dir=tmp_path, fname="test")
    assert recorder.fname == "test"

    # Invalid
    with pytest.raises(TypeError, match="'fname' must be an instance of"):
        StreamRecorder(record_dir=tmp_path, fname=101)


@requires_eeg_resting_state_dataset
def test_representation(tmp_path):
    """Test the representation method."""
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        recorder = StreamRecorder(record_dir=tmp_path, fname=None)
        expected = f"<Recorder: All streams | OFF | {tmp_path}>"
        assert recorder.__repr__() == expected
        recorder.start()
        expected = f"<Recorder: All streams | ON | {tmp_path}>"
        assert recorder.__repr__() == expected
        recorder.stop()
        expected = f"<Recorder: All streams | OFF | {tmp_path}>"
        assert recorder.__repr__() == expected

        recorder = StreamRecorder(
            record_dir=tmp_path, fname=None, stream_name="StreamPlayer"
        )
        expected = f"<Recorder: StreamPlayer | OFF | {tmp_path}>"
        assert recorder.__repr__() == expected
        recorder.start()
        expected = f"<Recorder: StreamPlayer | ON | {tmp_path}>"
        assert recorder.__repr__() == expected
        recorder.stop()
        expected = f"<Recorder: StreamPlayer | OFF | {tmp_path}>"
        assert recorder.__repr__() == expected
