"""
==========================================
StreamReceiver: real-time alpha band power
==========================================

BCI, Neurofeedback, or any online paradigm that needs access to real-time
signals to compute a given metric can be designed with a `~bsl.StreamReceiver`.
"""

#%%

# Authors: Mathieu Scheltienne <mathieu.scheltienne@gmail.com>
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
import os
import time
from pathlib import Path

import mne
import numpy as np
from matplotlib import pyplot as plt

from bsl import StreamRecorder, StreamReceiver, StreamPlayer, datasets
from bsl.utils import Timer
from bsl.triggers.software import TriggerSoftware

#%%
#
# To simulate an actual signal coming from an LSL stream, a `~bsl.StreamPlayer`
# is used with a 40 seconds resting-state recording.

stream_name = 'StreamPlayer'
fif_file = datasets.eeg_resting_state.data_path()
player = StreamPlayer(stream_name, fif_file)
player.start()
print (player)

#%%
# Basics of StreamReceiver
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that a `~bsl.StreamPlayer` is streaming data, a `~bsl.StreamReceiver` is
# defined to access it in real-time.

receiver = StreamReceiver(bufsize=2, winsize=1, stream_name='StreamPlayer')
time.sleep(2)  # wait 2 seconds to fill LSL inlet.

#%%%
#
# .. note::
#
#     A `~bsl.StreamReceiver` opens an LSL inlet for each connected stream at
#     initialization. The inlet's buffer is empty when created and fills up as
#     time passes. Data is pulled from the LSL inlet each time
#     `~bsl.StreamReceiver.acquire` is called.
#
# .. warning::
#
#     If the `~bsl.StreamReceiver` buffer/window is large and data is pulled
#     too often from the LSL inlet, there might not be enough new samples to
#     pull an entire window/buffer length.

receiver.acquire()
data1, timestamps1 = receiver.get_window()
print (data1.shape)
time.sleep(1)
receiver.acquire()
data2, timestamps2 = receiver.get_window()
print (data2.shape)
receiver.acquire()
data3, timestamps3 = receiver.get_window()
print (data3.shape)

#%%
#
# The code snippet above retrieved 3 different windows of 1 second each from
# the LSL stream sampled @ 512 Hz. The first window is retrieved 2 seconds
# after the `~bsl.StreamReceiver` was created. The second window is retrieved 1
# second after the second window. The third window is retrieved right after the
# second window.
#
# Let's visualize how this 3 different window are placed on the timeline:

idx = 10  # Select one channel
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(timestamps1, data1[:, idx], color='#1f77b4')
ax.plot(timestamps2, data2[:, idx], color='#ff7f0e')
ax.plot(timestamps3, data3[:, idx], color='#2ca02c')

#%%
#
# As expected, the second and third window are mostly overlapping and contains
# mostly the same data. To improve visualization, each window can be shifted
# vertically with a fix offset:

f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(timestamps1, data1[:, idx], color='#1f77b4')
ax.plot(timestamps2, data2[:, idx]+2, color='#ff7f0e')
ax.plot(timestamps3, data3[:, idx]+4, color='#2ca02c')

#%%
#
# Finally, the `~bsl.StreamReceiver.get_window` and
# `~bsl.StreamReceiver.get_buffer` methods are only getters and do not modify
# the buffer. Pulling new data in the buffer is only done in a separate Thread
# by `~bsl.StreamReceiver.acquire`. The exact timings at which the
# `~bsl.StreamReceiver` acquires new data is left to the discretion of the
# user.

data4, timestamps4 = receiver.get_window()
print ((data4 == data3).all(), (timestamps4 == timestamps3).all())

#%%
# Online loop with a StreamReceiver
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The example below is a simple online loop shifting between 2 phases lasting
# each 3 seconds. The loop will stop once each phase has been experienced
# twice, thus after 12 seconds. The beginning of each phase is marked with a
# trigger event, (1) or (2).
#
# As an example, the alpha band power will be computed for each phase on 1
# second successive acquisition window.

# FFT settings
winsize_in_samples = \
    receiver.streams['StreamPlayer'].sample_rate * receiver.winsize
sample_spacing = 1./receiver.streams['StreamPlayer'].sample_rate
frequencies = np.fft.rfftfreq(n=int(winsize_in_samples), d=sample_spacing)
alpha_band = np.where(np.logical_and(8<=frequencies, frequencies<=13))[0]
fft_window = np.hanning(winsize_in_samples)

# Loop settings
n_cycles = 2  # 2 alternation of phases
phase_duration = 3  # in seconds

#%%
#
# Acquired data is saved to disk with a `~bsl.StreamRecorder` and the beginning
# of each phase is marked with a trigger event. For this example, a
# `~bsl.triggers.software.TriggerSoftware` is used, but this example would be
# equally valid with a different type of trigger.
#
# .. note::
#
#     `~bsl.triggers.software.TriggerSoftware` must be created after a
#     `~bsl.StreamRecorder` is started and closed/deleted before a
#     `~bsl.StreamRecorder` is stopped.
#
#     .. code-block:: python
#
#         recorder = StreamRecorder()
#         recorder.start()
#         trigger = TriggerSoftware(recorder)
#         # do stuff
#         trigger.close() # OR >>> del trigger
#         recorder.stop()
#
#     All triggers do not need an active `~bsl.StreamRecorder` to be created.

record_dir = Path('~/bsl_data/examples').expanduser()
os.makedirs(record_dir, exist_ok=True)
recorder = StreamRecorder(record_dir, fname='example_real_time')
recorder.start()
print (recorder)
trigger = TriggerSoftware(recorder, verbose=True)

#%%
#
# The 2 first events, ``phase1`` and ``phase2`` are defined with their
# respective timings as tuples `(timing, event)`. The timing are offset by 0.2
# to give a little headroom to the script and to avoid clipping the first
# phase.
#
# The values used to mark the beginning of each phase are stored in a `dict`.

offset = 0.2  # offset to avoid clipping the first phase
events = [(offset, 'phase1'), (offset+phase_duration, 'phase2')]
trigger_values = {'phase1': 1, 'phase2': 2}

#%%
#
# There is actually 2 nested online loops: one to switch between phases and one
# to acquire data and compute the alpha band power inside a phase.

# number of time each phase has been experienced
n = 1

# list to store results
alphas = list()
timings = list()

# timers
paradigm_timer = Timer()  # timer used to switch between phases
phase_timer = Timer()  # timer used within a phase to count the duration

next_event_timing, event = events.pop(0)
while n <= n_cycles:
    if next_event_timing <= paradigm_timer.sec():
        # schedule next similar event
        events.append((next_event_timing+2*phase_duration, event))

        # add new result list
        alphas.append([])
        timings.append([])

        # reset timer and send trigger
        phase_timer.reset()
        trigger.signal(trigger_values[event])

        while phase_timer.sec() <= phase_duration:
            # acquisition
            receiver.acquire()
            raw, samples = receiver.get_window(return_raw=True)

            if samples.shape[0] != winsize_in_samples:
                continue  # skip incomplete windows

            # processing
            raw.set_eeg_reference(ref_channels='average', projection=False)
            data = raw.pick(picks='eeg', exclude='bads').get_data()
            data = np.multiply(data, fft_window)
            fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
            alpha = np.average(np.square(fftval[:, alpha_band]).T)

            # append to result list
            alphas[-1].append(alpha)
            timings[-1].append(samples[0])

        # increment if this is the second phase
        if event == 'phase2':
            n += 1
        # Retrieve next event
        next_event_timing, event = events.pop(0)

# close the trigger and stop the recorder
trigger.close()
recorder.stop()

#%%
#
# As you may have noticed, `~bsl.StreamReceiver.get_window` or
# `~bsl.StreamReceiver.get_buffer` return by default data as a `numpy.array`,
# but it can also be return directly as a `~mne.io.Raw` instance if the
# argument ``return_raw`` is set to ``True``.
#
# Depending on the CPU, on the current CPU load, and on the processing applied,
# the number of acquired window (points) may vary.

print ([len(a) for a in alphas])
print ([len(t) for t in timings])

# sphinx_gallery_thumbnail_number = 3
f, ax = plt.subplots(1, 1, figsize=(10, 10))
for k in range(len(alphas)):
    color = '#1f77b4' if k%2 == 0 else '#ff7f0e'
    ax.plot(timings[k], alphas[k], color=color)

#%%
#
# The saved `~mne.io.Raw` instance can then be loaded and analyzed.

fname = record_dir / 'fif' / 'example_real_time-StreamPlayer-raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
print (raw)
events = mne.find_events(raw, stim_channel='TRIGGER')
print (events)

#%%
#
# Stop the mock LSL stream.

del receiver  # disconnects and close the LSL inlet.
player.stop()
