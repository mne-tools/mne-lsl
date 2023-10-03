import math
import time

import mne
import numpy as np
import pytest

from bsl.datasets import testing
from bsl.lsl import StreamInlet, resolve_streams
from bsl.player import PlayerLSL as Player
from bsl.stream_viewer.scope import ScopeEEG
from bsl.stream_viewer.scope._scope import _BUFFER_DURATION


def test_scope_eeg():
    """Test EEG scope default capabilities."""
    stream_name = "StreamPlayer"

    with Player(testing.data_path() / "sample-eeg-ant-raw.fif", stream_name):
        streams = resolve_streams()
        inlet = StreamInlet(streams[0])
        inlet.open_stream()
        scope = ScopeEEG(inlet)

        # test init of BP filter
        assert not hasattr(scope, "_sos")
        assert not hasattr(scope, "_zi_coeff")
        assert not hasattr(scope, "_zi")
        scope.init_bandpass_filter(low=1.0, high=40.0)
        assert hasattr(scope, "_sos")
        assert hasattr(scope, "_zi_coeff")
        assert hasattr(scope, "_zi")
        assert scope._zi is None

        # test update loop
        assert len(scope._ts_list) == 0
        assert (scope._trigger_buffer == 0).all()
        assert (scope._data_buffer == 0).all()
        time.sleep(0.2)
        scope.update_loop()
        assert len(scope._ts_list) != 0
        assert (scope._data_buffer != 0).any()

        # car and BP were off
        assert scope._zi is None

        # test apply_car
        time.sleep(0.2)
        scope._apply_car = True
        scope.update_loop()

        # test apply BP
        time.sleep(0.2)
        scope._apply_bandpass = True
        scope.update_loop()
        assert scope._zi is not None

        # test change BP
        time.sleep(0.2)
        scope.init_bandpass_filter(low=1.0, high=20.0)
        assert scope._zi is None
        scope.update_loop()
        assert scope._zi is not None

        # test selection of channel
        time.sleep(0.2)
        n = len(scope._selected_channels)
        scope._selected_channels = scope._selected_channels[: n // 2]


def test_buffer_duration():
    """Test the buffer size."""
    stream_name = "StreamPlayer"
    raw = mne.io.read_raw_fif(
        testing.data_path() / "sample-eeg-ant-raw.fif", preload=False
    )
    sfreq = raw.info["sfreq"]

    with Player(testing.data_path() / "sample-eeg-ant-raw.fif", stream_name):
        streams = resolve_streams()
        inlet = StreamInlet(streams[0])
        inlet.open_stream()
        scope = ScopeEEG(inlet)
        assert scope.sample_rate == inlet.get_sinfo().sfreq

        assert scope.duration_buffer == _BUFFER_DURATION
        assert scope.duration_buffer_samples == math.ceil(_BUFFER_DURATION * sfreq)


def test_properties():
    """Test EEG scope properties."""
    stream_name = "StreamPlayer"
    raw = mne.io.read_raw_fif(
        testing.data_path() / "sample-eeg-ant-raw.fif", preload=False
    )

    with Player(testing.data_path() / "sample-eeg-ant-raw.fif", stream_name):
        streams = resolve_streams()
        inlet = StreamInlet(streams[0])
        inlet.open_stream()
        scope = ScopeEEG(inlet)

        assert scope.stream_name == scope._stream_name == stream_name
        assert scope.sample_rate == scope._sample_rate
        assert scope.duration_buffer == scope._duration_buffer
        assert scope.duration_buffer_samples == scope._duration_buffer_samples
        assert scope.ts_list == scope._ts_list == list()
        assert scope.channels_labels == scope._channels_labels == raw.ch_names[:-1]
        assert scope.nb_channels == scope._nb_channels == len(raw.ch_names[1:])
        assert scope.apply_car == scope._apply_car
        assert not scope.apply_car
        assert scope.apply_bandpass == scope._apply_bandpass
        assert not scope.apply_bandpass
        assert (
            scope.selected_channels
            == scope._selected_channels
            == list(range(scope.nb_channels))
        )
        assert (scope.data_buffer == scope._data_buffer).all()
        assert (scope.trigger_buffer == scope._trigger_buffer).all()

        with pytest.raises(AttributeError):
            scope.stream_name = "new name"
        with pytest.raises(AttributeError):
            scope.sample_rate = 101
        with pytest.raises(AttributeError):
            scope.duration_buffer = 101
        with pytest.raises(AttributeError):
            scope.duration_buffer_samples = 101
        with pytest.raises(AttributeError):
            scope.ts_list = [101]
        with pytest.raises(AttributeError):
            scope.channels_labels = ["101"]
        with pytest.raises(AttributeError):
            scope.nb_channels = 101
        scope.apply_car = True
        assert scope.apply_car
        scope.apply_bandpass = True
        assert scope.apply_bandpass
        scope.selected_channels = list(range(scope.nb_channels // 2))
        assert scope.selected_channels == list(range(scope.nb_channels // 2))
        with pytest.raises(AttributeError):
            scope.data_buffer = np.ones(
                (scope._nb_channels, scope._duration_buffer_samples),
                dtype=np.float32,
            )
        with pytest.raises(AttributeError):
            scope.trigger_buffer = np.ones(scope._duration_buffer_samples)
