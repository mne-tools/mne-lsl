import math
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


BP_ORDER = 2
_MAX_PLOT_DURATION = 30 # seconds

class _Scope:
    # ---------------------------- INIT ----------------------------
    def __init__(self, stream_receiver, stream_name):
        assert stream_name in stream_receiver.streams.keys()
        self.sr = stream_receiver
        self.stream_name = stream_name
        self.init_infos()
        self.init_arr(_MAX_PLOT_DURATION)

    def init_infos(self):
        self.sample_rate = int(
            self.sr.streams[self.stream_name].sample_rate)

    def init_arr(self, plot_duration):
        self.n_samples_buffer = math.ceil(plot_duration * self.sample_rate)
        self._ts_list = list()

    # -------------------------- Main Loop -------------------------
    def read_lsl_stream(self):
        self.sr.acquire()
        self.data_acquired, self._ts_list = self.sr.get_buffer()
        self.sr.reset_all_buffers()

        if len(self._ts_list) == 0:
            return

class _ScopeEEG(_Scope):
    # ---------------------------- INIT ----------------------------
    def __init__(self, stream_receiver, stream_name):
        super().__init__(stream_receiver, stream_name)
        self.init_signal_y_scales()
        self.init_variables()

    def init_infos(self):
        super().init_infos()
        self.channels_labels = self.sr.streams[self.stream_name].ch_list[1:]
        self.n_channels = len(
            self.sr.streams[self.stream_name].ch_list[1:])

    def init_signal_y_scales(self):
        self.signal_y_scales = {'1uV': 1, '10uV': 10, '25uV': 25,
                                '50uV': 50, '100uV': 100, '250uV': 250,
                                '500uV': 500, '1mV': 1000, '2.5mV': 2500,
                                '100mV': 100000}

    def init_variables(self):
        self._apply_car = False
        self._apply_bandpass = False
        self.channels_to_show_idx = list(range(self.n_channels))

    def init_arr(self, plot_duration):
        super().init_arr(plot_duration)
        self.trigger = np.zeros(self.n_samples_buffer)
        self.data_buffer = np.zeros((self.n_channels, self.n_samples_buffer),
                                    dtype=np.float32)

    def init_bandpass_filter(self, low, high):
        self.bp_low = low / (0.5 * self.sample_rate)
        self.bp_high = high / (0.5 * self.sample_rate)
        self.sos = butter(BP_ORDER, [self.bp_low, self.bp_high],
                          btype='band', output='sos')
        self.zi_coeff = sosfilt_zi(self.sos).reshape((self.sos.shape[0], 2, 1))
        self.zi = None

    # -------------------------- Main Loop -------------------------
    def update_loop(self):
        self.read_lsl_stream()
        if len(self._ts_list) > 0:
            self.filter_signal()
            # shape (channels, samples)
            self.data_buffer = np.roll(self.data_buffer, -len(self._ts_list),
                                       axis=1)
            self.data_buffer[:, -len(self._ts_list):] = self.data_acquired.T

    def read_lsl_stream(self):
        super().read_lsl_stream()
        # Remove trigger ch - shapes (samples, ) and (samples, channels)
        self.trigger = self.data_acquired[:, 0].reshape((-1, 1))
        self.data_acquired = self.data_acquired[:, 1:].reshape(
            (-1, self.n_channels))

    def filter_signal(self):
        if self._apply_bandpass:
            if self.zi is None:
                # Multiply by DC offset
                self.zi = self.zi_coeff * np.mean(self.data_acquired, axis=0)
            self.data_acquired, self.zi = sosfilt(
                self.sos, self.data_acquired, 0, self.zi)

        if self._apply_car:
            car_ch = np.mean(
                self.data_acquired[:, self.channels_to_show_idx], axis=1)
            self.data_acquired -= car_ch.reshape((-1, 1))
