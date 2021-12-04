import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ._scope import _Scope
from ...utils._logs import logger
from ...utils._docs import fill_doc, copy_doc
from ...utils import find_event_channel


BP_ORDER = 2


@fill_doc
class ScopeEEG(_Scope):
    """
    Class representing an EEG scope.

    Parameters
    ----------
    %(viewer_scope_stream_receiver)s
    %(viewer_scope_stream_name)s
    """

    # ---------------------------- INIT ----------------------------
    def __init__(self, stream_receiver, stream_name):
        super().__init__(stream_receiver, stream_name)

        # Infos from stream
        tch = find_event_channel(
            ch_names=self._sr.streams[self._stream_name].ch_list)
        # TODO: patch to be improved for multi-trig channel recording
        if isinstance(tch, list):
            tch = tch[0]
        assert tch is not None  # sanity-check

        self._channels_labels = \
            [channel for k, channel in enumerate(
                self._sr.streams[self._stream_name].ch_list) if k != tch]
        self._nb_channels = len(self._channels_labels)

        # Variables
        self._apply_car = False
        self._apply_bandpass = False
        self._selected_channels = list(range(self._nb_channels))

        # Buffers
        self._trigger_buffer = np.zeros(self._duration_buffer_samples)
        self._data_buffer = np.zeros(
            (self._nb_channels, self._duration_buffer_samples),
            dtype=np.float32)
        self._timestamps_buffer = np.zeros(self._duration_buffer_samples)

    def init_bandpass_filter(self, low, high):
        """
        Initialize the bandpass filter. The filter is a butter filter of order
        BP_ORDER (default 2).

        Parameters
        ----------
        low : int | float
            Frequency at which the signal is high-passed.
        high : int | float
            Frequency at which the signal is low-passed.
        """
        bp_low = low / (0.5 * self._sample_rate)
        bp_high = high / (0.5 * self._sample_rate)
        self._sos = butter(BP_ORDER, [bp_low, bp_high],
                           btype='band', output='sos')
        self._zi_coeff = sosfilt_zi(
            self._sos).reshape((self._sos.shape[0], 2, 1))
        self._zi = None

    # -------------------------- Main Loop -------------------------
    @copy_doc(_Scope.update_loop)
    def update_loop(self):
        self._read_lsl_stream()
        if len(self._ts_list) > 0:
            self._filter_signal()
            self._filter_trigger()
            # shape (channels, samples)
            self._data_buffer = np.roll(self._data_buffer, -len(self._ts_list),
                                        axis=1)
            self._data_buffer[:, -len(self._ts_list):] = self._data_acquired.T
            # shape (samples, )
            self._trigger_buffer = np.roll(
                self._trigger_buffer, -len(self._ts_list))
            self._trigger_buffer[-len(self._ts_list):] = self._trigger_acquired
            # shape (samples, )
            self._timestamps_buffer = np.roll(
                self._timestamps_buffer, -len(self._ts_list))
            self._timestamps_buffer[-len(self._ts_list):] = self._ts_list

    @copy_doc(_Scope._read_lsl_stream)
    def _read_lsl_stream(self):
        """
         The acquired data is splitted between the trigger channel and the data
        channels.
        """
        super()._read_lsl_stream()
        # Remove trigger ch - shapes (samples, ) and (samples, channels)
        self._trigger_acquired = self._data_acquired[:, 0]
        self._data_acquired = self._data_acquired[:, 1:].reshape(
            (-1, self._nb_channels))

    def _filter_signal(self):
        """
        Apply bandpass and CAR filter to the signal acquired if needed.
        """
        if self._apply_bandpass:
            if self._zi is None:
                logger.debug('Initialize ZI coefficient for BP.')
                # Multiply by DC offset
                self._zi = self._zi_coeff*np.mean(self._data_acquired, axis=0)
            self._data_acquired, self._zi = sosfilt(
                self._sos, self._data_acquired, 0, self._zi)

        if self._apply_car and len(self._selected_channels) >= 2:
            car_ch = np.mean(
                self._data_acquired[:, self._selected_channels], axis=1)
            self._data_acquired -= car_ch.reshape((-1, 1))

    def _filter_trigger(self, tol=0.05):
        """
        Cleans up the trigger signal by removing successive duplicates of a
        trigger value.
        """
        self._trigger_acquired[
            np.abs(np.diff(self._trigger_acquired, prepend=[0])) <= tol] = 0

    # --------------------------------------------------------------------
    @property
    def channels_labels(self):
        """
        List of the channel labels present in the connected stream.
        The TRIGGER channel is removed.
        """
        return self._channels_labels

    @property
    def nb_channels(self):
        """
        Number of channels present in the connected stream.
        The TRIGGER channel is removed.
        """
        return self._nb_channels

    @property
    def apply_car(self):
        """
        Boolean. Applies CAR if True.
        """
        return self._apply_car

    @apply_car.setter
    def apply_car(self, apply_car):
        self._apply_car = bool(apply_car)

    @property
    def apply_bandpass(self):
        """
        Boolean. Applies bandpass filter if True.
        """
        return self._apply_bandpass

    @apply_bandpass.setter
    def apply_bandpass(self, apply_bandpass):
        self._apply_bandpass = bool(apply_bandpass)

    @property
    def selected_channels(self):
        """
        List of indices of the selected channels.
        """
        return self._selected_channels

    @selected_channels.setter
    def selected_channels(self, selected_channels):
        self._selected_channels = selected_channels

    @property
    def data_buffer(self):
        """
        Data buffer (channels, samples).
        """
        return self._data_buffer

    @property
    def trigger_buffer(self):
        """
        Trigger buffer (samples, ).
        """
        return self._trigger_buffer
