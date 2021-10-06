"""
===================================
Real-time regulation of alpha waves
===================================
BCI, Neurofeedback, or any online paradigm that needs access to real-time
signals to compute a given metric can be achieved using a Stream Receiver.
This example will focus on a simple paradigm to regulate the alpha waves.
"""

# Authors: Mathieu Scheltienne <mathieu.scheltienne@gmail.com>
#
# License: LGPL-2.1

#%%
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from bsl import StreamRecorder, StreamReceiver, StreamPlayer, datasets
from bsl.triggers.software import TriggerSoftware
from bsl.utils import Timer

#%%
#
# Start a mock LSL stream with a Stream Player for this example purpose.
# Call in `__main__` because the Stream Player starts a new process, which can
# not be done outside `__main__` on Windows.
# See: https://docs.python.org/2/library/multiprocessing.html#windows

sample_data_raw_file = datasets.eeg_resting_state.data_path()
if __name__ == '__main__':
    player = StreamPlayer('MyStream', sample_data_raw_file)
    player.start()

#%%
#
# Define the name of the stream to connect to and the window size which will
# be retrieved and analyzed in real-time. Paradigms using multiple sync streams
# are possible but more difficult to design.
# Define the directory used by the recorder.

stream_name = 'MyStream'
window_size = 1  # in seconds
directory = Path('~/bsl_data/examples').expanduser()
os.makedirs(directory, exist_ok=True)

#%%
#
# Define a stream receiver and retrieves the window size in samples based on
# the stream info.
#
# Requires the stream player to be active, and thus, is called in `__main__`.

if __name__ == '__main__':
    receiver = StreamReceiver(
        bufsize=window_size, winsize=window_size, stream_name=stream_name)
    window_size_samples = receiver.streams[stream_name].buffer.winsize

#%%
#
# For the purpose of this example, we will assume a paradigm where 2 phases,
# rest (1) and regulation (2), of 5 seconds each, alternates 3 times. During
# the regulation phase, the subject is asked to try to regulate his alpha
# waves. The measured alpha waves could be displayed on a screen for a
# neurofeedback study.
#
# Define the variables required for computing the alpha power.
#
# Requires the stream player to be active, and thus, is called in `__main__`.

if __name__ == '__main__':
    frequencies = np.fft.rfftfreq(
        n=window_size_samples,
        d=1./receiver.streams[stream_name].sample_rate)
    alpha_band = np.where(np.logical_and(8<=frequencies, frequencies<=13))[0]
    fft_window = np.hanning(window_size_samples)

#%%
#
# Define the settings of the online paradigm loop.

n_cycles = 2  # 2 alternation of rest/regulation phases
phase_duration = 2  # in seconds
paradigm_timer = Timer()  # timer used to switch between phases
phase_timer = Timer()  # timer used within a phase to count the duration

#%%

# The 2 first events are defined with their respective timings as tuples
# `(timing, event)`. The timing are offset by 1 to give a little headroom to
# the script and to avoid clipping the first phase.

offset = 0.2  # offset to avoid clipping the first phase
events = [(offset, 'rest'), (offset+phase_duration, 'regulation')]

#%%
#
# Define the value which will be used by the trigger to mark the beginning of
# each phase.

trigger_values = {'rest': 1, 'regulation': 2}

#%%
#
# Define in a function the paradigm.
# - Define a recorder and start it.
# - Define a trigger.
# - Define the paradigm loop.
# - Close the trigger and stop the recorder.

def my_online_paradigm():
    """
    Function called in __main__.
    """
    # Define a recorder and start it
    recorder = StreamRecorder(directory)
    recorder.start(fif_subdir=False, verbose=False)
    # Define a trigger
    trigger = TriggerSoftware(recorder=recorder, verbose=True)

    # Define loop counter and output lists
    n = 1
    alphas = list()
    timings = list()

    # Paradigmn loop
    next_event_timing, event = events.pop(0)
    paradigm_timer.reset()
    while n <= n_cycles:
        if next_event_timing <= paradigm_timer.sec():
            # schedule next similar event
            events.append((next_event_timing+2*phase_duration, event))

            # add list to the ouputs
            alphas.append([])
            timings.append([])

            # reset timer and send trigger
            phase_timer.reset()
            trigger.signal(trigger_values[event])

            while phase_timer.sec() <= phase_duration:
                # acquisition
                receiver.acquire()
                raw, samples = receiver.get_window(return_raw=True)

                if samples.shape[0] != window_size_samples:
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

            # Increment if this is a regulation phase
            if event == 'regulation':
                n += 1
            # Retrieve next event
            next_event_timing, event = events.pop(0)

    # Close the trigger and stop the recorder
    trigger.close()
    recorder.stop()

    return alphas, timings

#%%
#
# Call in `__main__` the function. The StreamRecorder starts a new process,
# which can not be done outside `__main__` on Windows.
# See: https://docs.python.org/2/library/multiprocessing.html#windows

if __name__ == '__main__':
    alphas, timings = my_online_paradigm()

#%%
#
# Stop the mock LSL stream.

if __name__ == '__main__':
    del receiver
    player.stop()

#%%
#
# Depending on the CPU, on the current CPU load, on the processing apply, the
# number of acquired window (points) may vary.

if __name__ == '__main__':
    print ([len(a) for a in alphas])
    print ([len(t) for t in timings])

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    for k in range(len(alphas)):
        color = 'lightblue' if k%2 == 0 else 'teal'
        ax.plot(timings[k], alphas[k], color=color)
