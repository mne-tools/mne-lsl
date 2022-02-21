"""
==============================================================
StreamReceiver: real-time buffer filtered with a causal filter
==============================================================

A `~bsl.StreamReceiver` can be used to create a data buffer on which different
operations have already been applied. For instance, a buffer where data is
filtered with a bandpass filter.
"""

#%%

# Authors: Mathieu Scheltienne <mathieu.scheltienne@fcbg.ch>
#
# License: LGPL-2.1

#%%
# .. warning::
#
#     Both `~bsl.StreamPlayer` and `~bsl.StreamRecorder` create a new process
#     to stream or record data. On Windows, mutliprocessing suffers a couple of
#     restrictions. The entry-point of a multiprocessing program should be
#     protected with ``if __name__ == '__main__':`` to ensure it can safely
#     import and run the module. More information on the
#     `documentation for multiprocessing on Windows
#     <https://docs.python.org/2/library/multiprocessing.html#windows>`_.
#
# This example will use a sample EEG resting-state dataset that can be retrieve
# with :ref:`bsl.datasets<datasets>`. The dataset is stored in the user home
# directory in the folder ``bsl_data`` (e.g. ``C:\Users\User\bsl_data``).

#%%
from math import ceil
import time

from matplotlib import pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from bsl import StreamReceiver, StreamPlayer, datasets
from bsl.utils import Timer

#%%
#
# To simulate an actual signal coming from an LSL stream, a `~bsl.StreamPlayer`
# is used with a 40 seconds resting-state recording. This dataset is already
# filtered between (1, 40) Hz.

stream_name = 'StreamPlayer'
fif_file = datasets.eeg_resting_state.data_path()
player = StreamPlayer(stream_name, fif_file)
player.start()
print (player)

#%%

raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
print (f"BP filter between: {raw.info['highpass']}, {raw.info['lowpass']} Hz")

#%%
# Filter
# ^^^^^^
#
# Data should be filtered along one dimension. For this example, a butter IIR
# filter is used. More information on filtering is available on the MNE
# documentation:
#
# - `Background information on filtering <https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html#disc-filtering>`_
# - `Filtering and resampling data <https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html#tut-filter-resample>`_

def create_bandpass_filter(low, high, fs, n):
    """
    Create a bandpass filter using a butter filter of order n.

    Parameters
    ----------
    low : float
        The lower pass-band edge.
    high : float
        The upper pass-band ege.
    fs : float
        Sampling rate of the data.
    n : int
        Order of the filter.

    Returns
    -------
    sos : array
        Second-order sections representation of the IIR filter.
    zi_coeff : array
        Initial condition for sosfilt for step response steady-state.
    """
    # Divide by the Nyquist frequency
    bp_low = low / (0.5 * fs)
    bp_high = high / (0.5 * fs)
    # Compute SOS output (second order sections)
    sos = butter(n, [bp_low, bp_high], btype='band', output='sos')
    # Construct initial conditions for sosfilt for step response steady-state.
    zi_coeff = sosfilt_zi(sos).reshape((sos.shape[0], 2, 1))

    return sos, zi_coeff

#%%
#
# EEG data is usually subject to a lage DC offset, which corresponds to a step
# response steady-state. The initial conditions are determined by multiplying
# the ``zi_coeff`` with the DC offset. The DC offset value can be approximated
# by taking the mean of a small window.

#%%
# Buffer
# ^^^^^^^
#
# When creating the filtered buffer, the duration has to be define to create a
# numpy array of the correct shape and pre-allocate the required space.

buffer_duration = 5  # in seconds

#%%
#
# Then, a `~bsl.StreamReceiver` is created. But the actual buffer and window
# size of the `~bsl.StreamReceiver` are set as small as possible. The buffer
# from the `~bsl.StreamReceiver` is only used to store the last samples until
# they are retrieved, filtered, and added to the filtered buffer. In this
# example, the `~bsl.StreamReceiver` buffer and window size are set to 200 ms.

sr = StreamReceiver(bufsize=0.2, winsize=0.2, stream_name=stream_name)
time.sleep(0.2)  # wait to fill LSL inlet.

#%%%
#
# .. note::
#
#     A `~bsl.StreamReceiver` opens an LSL inlet for each connected stream at
#     initialization. The inlet's buffer is empty when created and fills up as
#     time passes. Data is pulled from the LSL inlet each time
#     `~bsl.StreamReceiver.acquire` is called.
#
# The filtered buffer can be define as a class that uses the elements created
# previously. The method ``.update()`` pulls new samples from the LSL stream,
# filters and add them to the buffer while removing older samples that are now
# exiting the buffer.
#
# The `~bsl.StreamReceiver` appends a ``TRIGGER`` channel at the beginning of
# the data array. For filtering, the trigger channel is not needed. Thus, the
# number of channels is reduced by 1 and the first channel on the
# ``(samples, channels)`` data array is ignored.
#
# For this example, the filter is defined between 5 and 10 Hz to emphasize its
# effect as the dataset streamed is already filtered between (1, 40) Hz.

class Buffer:
    """
    A buffer containing filter data and its associated timestamps.

    Parameters
    ----------
    buffer_duration : float
        Length of the buffer in seconds.
    sr : bsl.StreamReceiver
        StreamReceiver connected to the desired data stream.
    """

    def __init__(self, buffer_duration, sr):
        # Store the StreamReceiver in a class attribute
        self.sr = sr

        # Retrieve sampling rate and number of channels
        self.fs = int(self.sr.streams[stream_name].sample_rate)
        self.nb_channels = len(self.sr.streams[stream_name].ch_list) - 1

        # Define duration
        self.buffer_duration = buffer_duration
        self.buffer_duration_samples = ceil(self.buffer_duration * self.fs)

        # Create data array
        self.timestamps = np.zeros(self.buffer_duration_samples)
        self.data = np.zeros((self.buffer_duration_samples, self.nb_channels))
        # For demo purposes, let's store also the raw data
        self.raw_data = np.zeros((self.buffer_duration_samples,
                                  self.nb_channels))

        # Create filter BP (1, 15) Hz and filter variables
        self.sos, self.zi_coeff = create_bandpass_filter(5., 10., self.fs, n=2)
        self.zi = None

    def update(self):
        """
        Update the buffer with new samples from the StreamReceiver. This method
        should be called regularly, with a period at least smaller than the
        StreamReceiver buffer length.
        """
        # Acquire new data points
        self.sr.acquire()
        data_acquired, ts_list = self.sr.get_buffer()
        self.sr.reset_buffer()

        if len(ts_list) == 0:
            return  # break early, no new samples

        # Remove trigger channel
        data_acquired = data_acquired[:, 1:]

        # Filter acquired data
        if self.zi is None:
            # Initialize the initial conditions for the cascaded filter delays.
            self.zi = self.zi_coeff * np.mean(data_acquired, axis=0)
        data_filtered, self.zi = sosfilt(self.sos, data_acquired, axis=0,
                                         zi=self.zi)

        # Roll buffer, remove samples exiting and add new samples
        self.timestamps = np.roll(self.timestamps, -len(ts_list))
        self.timestamps[-len(ts_list):] = ts_list
        self.data = np.roll(self.data, -len(ts_list), axis=0)
        self.data[-len(ts_list):, :] = data_filtered
        self.raw_data = np.roll(self.raw_data, -len(ts_list), axis=0)
        self.raw_data[-len(ts_list):, :] = data_acquired

#%%
# Testing the filtered buffer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The filtered buffer must be updated regularly. In this example, the
# `~bsl.StreamReceiver` buffer has been initialized at 200 ms. Thus, the
# filtered buffer should be updated at most every 200 ms, else, there is a risk
# that a couple of samples will be missed between 2 updates.
#
# A 15 seconds acquisition is used to test the buffer. Every 5 seconds, the
# buffer is retrieved and plotted.

# Create plot
f, ax = plt.subplots(2, 1, sharex=True)
ax[0].set_title('Raw data')
ax[1].set_title('Filtered data between (5, 10) Hz')
# Create buffer
buffer = Buffer(buffer_duration, sr)

# Start by filling once the entire buffer (to get rid of the initialization 0)
timer = Timer()
while timer.sec() <= buffer_duration:
    buffer.update()

# Acquire during 15 seconds and plot every 5 seconds
idx_last_plot = 1
timer.reset()
while timer.sec() <= 15:
    buffer.update()
    # check if we just passed the 5s between plot limit
    if timer.sec() // 5 == idx_last_plot:
        # average all channels to simulate an evoked response
        ax[0].plot(buffer.timestamps, np.mean(buffer.raw_data[:, 1:], axis=1))
        ax[1].plot(buffer.timestamps, np.mean(buffer.data[:, 1:], axis=1))
        idx_last_plot += 1

#%%
#
# Stop the mock LSL stream.

del sr  # disconnects and close the LSL inlet.
player.stop()
