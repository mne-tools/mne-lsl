import math
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


BP_ORDER = 2
_BUFFER_DURATION = 30 # seconds

class _Scope:
    """
    Class representing a base scope.

    Parameters
    ----------
    stream_receiver : neurodecode.stream_receiver.StreamReceiver
        The connected stream receiver, which can be connected to multiple streams.
    stream_name : str
        The stream to connect to.
    """
    # ---------------------------- INIT ----------------------------
    def __init__(self, stream_receiver, stream_name):
        assert stream_name in stream_receiver.streams.keys()
        self.sr = stream_receiver
        self.stream_name = stream_name
        self.init_infos()
        self.init_buffer(_BUFFER_DURATION)

    def init_infos(self):
        """
        Extract basic stream informations.
        """
        self.sample_rate = int(
            self.sr.streams[self.stream_name].sample_rate)

    def init_buffer(self, duration_buffer):
        """
        Initialize buffer(s).
        """
        self.duration_buffer = duration_buffer
        self.n_samples_buffer = math.ceil(duration_buffer * self.sample_rate)
        self._ts_list = list()

    # -------------------------- Main Loop -------------------------
    def read_lsl_stream(self):
        """
        Acquires data from the connected LSL stream.
        """
        self.sr.acquire()
        self.data_acquired, self._ts_list = self.sr.get_buffer()
        self.sr.reset_all_buffers()

        if len(self._ts_list) == 0:
            return

class _ScopeEEG(_Scope):
    """
    Class representing an EEG scope.

    Parameters
    ----------
    stream_receiver : neurodecode.stream_receiver.StreamReceiver
        The connected stream receiver, which can be connected to multiple streams.
    stream_name : str
        The stream to connect to.
    """
    # ---------------------------- INIT ----------------------------
    def __init__(self, stream_receiver, stream_name):
        super().__init__(stream_receiver, stream_name)
        self.init_signal_y_scales()
        self.init_variables()

    def init_infos(self):
        """
        Extract basic stream informations.
        """
        super().init_infos()
        self.channels_labels = self.sr.streams[self.stream_name].ch_list[1:]
        self.n_channels = len(
            self.sr.streams[self.stream_name].ch_list[1:])

    def init_signal_y_scales(self):
        """
        The available signal scale/range values as a dictionnary {key: value}
        with key a representative string and value in uV.
        """
        self.signal_y_scales = {'1uV': 1, '10uV': 10, '25uV': 25,
                                '50uV': 50, '100uV': 100, '250uV': 250,
                                '500uV': 500, '1mV': 1000, '2.5mV': 2500,
                                '100mV': 100000}

    def init_variables(self):
        """
        Initialize variables.
        """
        self._apply_car = False
        self._apply_bandpass = False
        self.channels_to_show_idx = list(range(self.n_channels))

    def init_buffer(self, plot_duration):
        """
        Initialize buffer(s).
        """
        super().init_buffer(plot_duration)
        self.trigger_buffer = np.zeros(self.n_samples_buffer)
        self.data_buffer = np.zeros((self.n_channels, self.n_samples_buffer),
                                    dtype=np.float32)

    def init_bandpass_filter(self, low, high):
        """
        Initialize the bandpass filter. The filter is a butter filter of order
        neurodecode.stream_viewer._scope.BP_ORDER

        Parameters
        ----------
        low : int | float
            The frequency at which the signal is high-passed.
        high : int | float
            The frequency at which the signal is low-passed.
        """
        self.bp_low = low / (0.5 * self.sample_rate)
        self.bp_high = high / (0.5 * self.sample_rate)
        self.sos = butter(BP_ORDER, [self.bp_low, self.bp_high],
                          btype='band', output='sos')
        self.zi_coeff = sosfilt_zi(self.sos).reshape((self.sos.shape[0], 2, 1))
        self.zi = None

    # -------------------------- Main Loop -------------------------
    def update_loop(self):
        """
        Main update loop acquiring data from the LSL stream and filling the
        scope's buffer.
        """
        self.read_lsl_stream()
        if len(self._ts_list) > 0:
            self.filter_signal()
            self.filter_trigger()
            # shape (channels, samples)
            self.data_buffer = np.roll(self.data_buffer, -len(self._ts_list),
                                       axis=1)
            self.data_buffer[:, -len(self._ts_list):] = self.data_acquired.T
            # shape (samples, )
            self.trigger_buffer = np.roll(self.trigger_buffer, -len(self._ts_list))
            self.trigger_buffer[-len(self._ts_list):] = self.trigger_acquired

    def read_lsl_stream(self):
        """
        Acquires data from the connected LSL stream. The acquired data is
        splitted between the trigger channel and the data channels.
        """
        super().read_lsl_stream()
        # Remove trigger ch - shapes (samples, ) and (samples, channels)
        self.trigger_acquired = self.data_acquired[:, 0]
        self.data_acquired = self.data_acquired[:, 1:].reshape(
            (-1, self.n_channels))

    def filter_signal(self):
        """
        Apply bandpass and CAR filter to the signal acquired if needed.
        """
        if self._apply_bandpass:
            if self.zi is None:
                # Multiply by DC offset
                self.zi = self.zi_coeff * np.mean(self.data_acquired, axis=0)
            self.data_acquired, self.zi = sosfilt(
                self.sos, self.data_acquired, 0, self.zi)

        if self._apply_car and len(self.channels_to_show_idx) >= 2:
            car_ch = np.mean(
                self.data_acquired[:, self.channels_to_show_idx], axis=1)
            self.data_acquired -= car_ch.reshape((-1, 1))

    def filter_trigger(self, tol=0.05):
        """
        Cleans up the trigger signal by removing successive duplicates of a
        trigger value.
        """
        self.trigger_acquired[
            np.abs(np.diff(self.trigger_acquired, prepend=[0])) <= tol] = 0
