import time
import pickle
from pathlib import Path

import mne

from bsl import StreamRecorder
from bsl.datasets import sample
from bsl.utils._testing import Stream, requires_sample_dataset


def _check_recorder_fname_exists(record_dir, eve_file, stream, fif_subdir):
    """Use eve_file to retrieve the file name stem and check if the recorder
    files exists."""
    fname_stem = eve_file.stem.split('-eve')[0]
    fname_pcl = record_dir / f'{fname_stem}-{stream}-raw.pcl'
    if fif_subdir:
        fname_fif = record_dir / 'fif' / f'{fname_stem}-{stream}-raw.fif'
    else:
        fname_fif = record_dir / f'{fname_stem}-{stream}-raw.fif'
    assert fname_pcl.exists()
    assert fname_fif.exists()


def _check_recorder_files(dataset, record_duration, record_dir, eve_file,
                          stream, fif_subdir):
    """Checks the recorded files content."""
    fname_stem = eve_file.stem.split('-eve')[0]
    fname_pcl = record_dir / f'{fname_stem}-{stream}-raw.pcl'
    if fif_subdir:
        fname_fif = record_dir / 'fif' / f'{fname_stem}-{stream}-raw.fif'
    else:
        fname_fif = record_dir / f'{fname_stem}-{stream}-raw.fif'
    raw = mne.io.read_raw_fif(fname=dataset.data_path(), preload=True)
    with open(fname_pcl, 'rb') as inp:
        raw_pcl = pickle.load(inp)
    raw_fif = mne.io.read_raw_fif(fname_fif, preload=True)
    assert raw.ch_names == raw_pcl['ch_names'] == raw_fif.ch_names
    assert raw.info['sfreq'] == raw_pcl['sample_rate'] == raw_fif.info['sfreq']
    assert raw_pcl['signals'].shape[::-1] == raw_fif.get_data().shape
    # some delay is introduce by the process operations
    assert 0 <= raw_fif.n_times / raw_fif.info['sfreq'] - record_duration < 0.2


@requires_sample_dataset
def test_recording(tmp_path):
    """Test recording capability of the stream recorder."""
    stream = 'StreamPlayer'
    record_duration = 2  # seconds
    dataset = sample
    with Stream(stream, dataset):
        recorder = StreamRecorder(record_dir=tmp_path)
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()

    _check_recorder_fname_exists(tmp_path, eve_file, stream, fif_subdir=False)
    _check_recorder_files(dataset, record_duration, tmp_path,
                          eve_file, stream, fif_subdir=False)


@requires_sample_dataset
def test_recording_multiple_streams(tmp_path):
    """Test multi-stream recording capabilities of the stream recorder."""
    record_duration = 2  # seconds
    dataset = sample
    with Stream('StreamPlayer1', dataset), Stream('StreamPlayer2', dataset):
        # Record only StreamPlayer1
        recorder = StreamRecorder(record_dir=tmp_path / 'only1',
                                  stream_name='StreamPlayer1')
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorder_fname_exists(tmp_path / 'only1', eve_file,
                                     'StreamPlayer1', fif_subdir=False)
        _check_recorder_files(dataset, record_duration, tmp_path / 'only1',
                              eve_file, 'StreamPlayer1', fif_subdir=False)

        # Record only StreamPlayer2
        recorder.stream_name = 'StreamPlayer2'
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorder_fname_exists(tmp_path / 'only1', eve_file,
                                     'StreamPlayer2', fif_subdir=False)
        _check_recorder_files(dataset, record_duration, tmp_path / 'only1',
                              eve_file, 'StreamPlayer2', fif_subdir=False)

        # Record both
        recorder.record_dir = tmp_path
        recorder.stream_name = None
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorder_fname_exists(tmp_path, eve_file,
                                     'StreamPlayer1', fif_subdir=False)
        _check_recorder_fname_exists(tmp_path, eve_file,
                                     'StreamPlayer2', fif_subdir=False)
        _check_recorder_files(dataset, record_duration, tmp_path,
                              eve_file, 'StreamPlayer1', fif_subdir=False)
        _check_recorder_files(dataset, record_duration, tmp_path,
                              eve_file, 'StreamPlayer2', fif_subdir=False)


@requires_sample_dataset
def test_property_setter(tmp_path):
    """Test changing the properties before and during an on-going recording."""
    stream = 'StreamPlayer'
    with Stream(stream, sample):
        recorder = StreamRecorder(record_dir=Path.cwd())
        # Before recording start
        assert recorder.record_dir == Path.cwd()
        assert recorder.fname is None
        assert recorder.stream_name is None
        recorder.record_dir = tmp_path
        recorder.fname = 'test'
        recorder.stream_name = stream

        # Start
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        # Should fail to change anything
        recorder.record_dir = Path.cwd()
        recorder.fname = 'blabla'
        recorder.stream_name = 'imaginary stream'
        assert recorder.record_dir == tmp_path
        assert recorder.fname == 'test'
        assert recorder.stream_name == stream
        # Stop
        eve_file = recorder.eve_file
        time.sleep(2)
        recorder.stop()
        # Checks
        assert eve_file.stem.split('-eve')[0] == 'test'
        _check_recorder_fname_exists(tmp_path, eve_file, stream,
                                     fif_subdir=False)

        # Change values
        recorder.record_dir = tmp_path / 'test'
        recorder.fname = None
        recorder.stream_name = None
        # Start
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        # Stop
        eve_file = recorder.eve_file
        time.sleep(2)
        recorder.stop()

        # Checks
        assert eve_file.stem.split('-eve')[0] != 'test'
        _check_recorder_fname_exists(tmp_path / 'test', eve_file, stream,
                                     fif_subdir=False)


@requires_sample_dataset
def test_arg_fif_subdir(tmp_path):
    """Test argument fif_subdir."""
    record_duration = 2  # seconds
    dataset = sample
    with Stream('StreamPlayer', dataset):
        recorder = StreamRecorder(record_dir=tmp_path)
        recorder.start(fif_subdir=False, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorder_fname_exists(tmp_path, eve_file,
                                     'StreamPlayer', fif_subdir=False)
        _check_recorder_files(dataset, record_duration, tmp_path,
                              eve_file, 'StreamPlayer', fif_subdir=False)

        recorder.start(fif_subdir=True, blocking=True, verbose=False)
        eve_file = recorder.eve_file
        time.sleep(record_duration)
        recorder.stop()
        _check_recorder_fname_exists(tmp_path, eve_file,
                                     'StreamPlayer', fif_subdir=True)
        _check_recorder_files(dataset, record_duration, tmp_path,
                              eve_file, 'StreamPlayer', fif_subdir=True)
