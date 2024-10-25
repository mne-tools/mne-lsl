import math

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ...utils._docs import copy_doc
from ...utils.logs import logger
from ._scope import _Scope

BP_ORDER = 2


class ScopeEEG(_Scope):
    """Class representing an EEG scope.

    Parameters
    ----------
    inlet : StreamInlet
    """

    # ---------------------------- INIT ----------------------------
    def __init__(self, inlet):
        super().__init__(inlet)

        self._info = self._inlet.get_sinfo().get_channel_info()
        self._channels_labels = self._info.ch_names
        self._picks = list(range(len(self._channels_labels)))
        # patch for CB classic trigger channel on ANT devices:
        if inlet.name.startswith("eegoSports"):
            if inlet.name == "eegoSports 0000" and "176" in self._channels_labels:
                self._tch_label = "176"
                self._tch = self._channels_labels.index(self._tch_label)
                self._channels_labels.pop(self._tch)
                self._picks.pop(self._tch)
            elif "TRIGGER" in self._channels_labels:  # 64 channels TRG
                self._tch_label = "TRIGGER"
                self._tch = self._channels_labels.index(self._tch_label)
                self._channels_labels.pop(self._tch)
                self._picks.pop(self._tch)
        else:
            self._tch_label = None
            self._tch = None
        self._picks = np.array(self._picks)
        self._nb_channels = len(self._channels_labels)

        # Variables
        self._apply_bandpass = False
        self._apply_car = False
        self._apply_detrend = False
        self._detrend_mean = None
        self._selected_channels = list(range(self._nb_channels))

        # Buffers
        self._trigger_buffer = np.zeros(self._duration_buffer_samples)
        self._data_buffer = np.zeros(
            (self._nb_channels, self._duration_buffer_samples),
            dtype=np.float32,
        )

    def init_bandpass_filter(self, low, high):
        """Initialize the bandpass filter.

        The filter is a butter filter of order BP_ORDER (default 2).

        Parameters
        ----------
        low : int | float
            Frequency at which the signal is high-passed.
        high : int | float
            Frequency at which the signal is low-passed.
        """
        logger.debug("Bandpass initialization for [%s, %s] Hz..", low, high)
        bp_low = low / (0.5 * self._sample_rate)
        bp_high = high / (0.5 * self._sample_rate)
        self._sos = butter(BP_ORDER, [bp_low, bp_high], btype="band", output="sos")
        self._zi_coeff = sosfilt_zi(self._sos).reshape((self._sos.shape[0], 2, 1))
        self._zi = None
        logger.debug("Bandpass initialization complete.")

    # -------------------------- Main Loop -------------------------
    @copy_doc(_Scope.update_loop)
    def update_loop(self):
        self._read_lsl_stream()
        if len(self._ts_list) > 0:
            self._filter_signal()
            if self._tch is not None:
                self._filter_trigger()
            # shape (channels, samples)
            self._data_buffer = np.roll(self._data_buffer, -len(self._ts_list), axis=1)
            self._data_buffer[:, -len(self._ts_list) :] = self._data_acquired.T
            # shape (samples, )
            if self._tch is not None:
                self._trigger_buffer = np.roll(
                    self._trigger_buffer, -len(self._ts_list)
                )
                self._trigger_buffer[-len(self._ts_list) :] = self._trigger_acquired

    @copy_doc(_Scope._read_lsl_stream)
    def _read_lsl_stream(self):
        """
         The acquired data is split between the trigger channel and the data
        channels.
        """
        super()._read_lsl_stream()
        # Remove trigger ch - shapes (samples, ) and (samples, channels)
        if self._tch is not None:
            self._trigger_acquired = self._data_acquired[:, self._tch]
        self._data_acquired = self._data_acquired[:, self._picks].reshape(
            (-1, self._nb_channels)
        )

    def _filter_signal(self):
        """Apply bandpass and CAR filter to the signal acquired if needed."""
        if self._apply_detrend:
            if self._detrend_mean is None:
                raise RuntimeError(
                    "The variable _detrend_mean should not be None if "
                    "the detrending checkbox is ticked. Please contact a developer. "
                )
            # shape (channels, samples)
            self._detrend_mean = np.roll(
                self._detrend_mean, -len(self._ts_list), axis=1
            )
            self._detrend_mean[:, -len(self._ts_list) :] = self._data_acquired.T
            if not np.all(self._detrend_mean[:, 0] == 0):
                self._data_acquired -= np.mean(self._detrend_mean, axis=1)

        if self._apply_bandpass:
            if self._zi is None:
                logger.debug("Initialize ZI coefficient for BP.")
                # Multiply by DC offset
                self._zi = self._zi_coeff * np.mean(self._data_acquired, axis=0)
            self._data_acquired, self._zi = sosfilt(
                self._sos, self._data_acquired, 0, self._zi
            )

        if self._apply_car and len(self._selected_channels) >= 2:
            car_ch = np.mean(self._data_acquired[:, self._selected_channels], axis=1)
            self._data_acquired -= car_ch.reshape((-1, 1))

    def _filter_trigger(self, tol=0.05):  # noqa
        """Remove successive duplicates of a trigger value."""
        if self._tch_label == "176":
            mask = np.ones(self._trigger_acquired.size, dtype=bool)
            idx = np.where(self._trigger_acquired == 128)[0]
            mask[idx] = False
            self._trigger_acquired = self._trigger_acquired * mask
        self._trigger_acquired[
            np.abs(np.diff(self._trigger_acquired, prepend=[0])) <= tol
        ] = 0

    # --------------------------------------------------------------------
    @property
    def channels_labels(self):
        """List of the channel labels present in the connected stream.

        The TRIGGER channel is removed.
        """
        return self._channels_labels

    @property
    def nb_channels(self):
        """Number of channels present in the connected stream.

        The TRIGGER channel is removed.
        """
        return self._nb_channels

    @property
    def apply_bandpass(self):
        """Boolean. Applies bandpass filter if True."""
        return self._apply_bandpass

    @apply_bandpass.setter
    def apply_bandpass(self, apply_bandpass):
        self._apply_bandpass = bool(apply_bandpass)

    @property
    def apply_car(self):
        """Boolean. Applies CAR if True."""
        return self._apply_car

    @apply_car.setter
    def apply_car(self, apply_car):
        self._apply_car = bool(apply_car)

    @property
    def apply_detrend(self):
        """Boolean. Applies detrending if True."""
        return self._apply_detrend

    @apply_detrend.setter
    def apply_detrend(self, apply_detrend):
        self._apply_detrend = bool(apply_detrend)
        if self._apply_detrend is True:
            self._detrend_mean = np.zeros(
                (self._nb_channels, math.ceil(2 * self._sample_rate)),
                dtype=np.float32,
            )
        else:
            self._detrend_mean = None

    @property
    def selected_channels(self):
        """List of indices of the selected channels."""
        return self._selected_channels

    @selected_channels.setter
    def selected_channels(self, selected_channels):
        self._selected_channels = selected_channels

    @property
    def data_buffer(self):
        """Data buffer (channels, samples)."""
        return self._data_buffer

    @property
    def trigger_buffer(self):
        """Trigger buffer (samples, )."""
        return self._trigger_buffer
