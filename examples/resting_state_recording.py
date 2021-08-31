"""
=======================
Resting-State recording
=======================
A resting-state recording is a simple offline recording during which the brain
activity of a subject is measured in the absence of any stimulus or task.
"""

# Authors: Mathieu Scheltienne <mathieu.scheltienne@gmail.com>
#
# License: LGPL-2.1

#%%

import time
import datetime
from pathlib import Path

from bsl import StreamRecorder, StreamPlayer, datasets
from bsl.triggers.software import TriggerSoftware
from bsl.utils import Timer
from bsl.utils.io._file_dir import make_dirs

#%%
#
# Start a mock LSL stream with a Stream Player for this example purpose.
# Call in `__main__` because the Stream Player starts a new process, which can
# not be done outside `__main__` on Windows.
# See: https://docs.python.org/2/library/multiprocessing.html#windows

sample_data_raw_file = datasets.sample.data_path()
if __name__ == '__main__':
    player = StreamPlayer('StreamPlayer', sample_data_raw_file)
    player.start()

#%%
#
# Define the directory used by the recorder and the name of the streams to
# connect to. Without specifying the argument `stream_name`, the recorder will
# connect to every LSL stream available and save one .pcl and one .fif file per
# stream.

directory = Path('~/bsl_data/examples').expanduser()
make_dirs(directory)
stream_name = None

#%%
#
# Define the duration of the resting-state recording in seconds.
# Typical resting-state recordings last several minutes.

duration = 5

#%%
#
# Define in a function the paradigm.
# - Define a recorder and start it.
# - Define a trigger.
# - Send a signal on the trigger and wait for the duration.
# - Close the trigger and stop the recorder.
#
# The software trigger used in this example requires an active stream recorder
# to link to. Other types of trigger may be used, may be defined before the
# recorder, and may not need to be closed.

def resting_state(directory, stream_name, duration):
    """
    Function called in __main__.
    """
    recorder = StreamRecorder(directory)
    recorder.start(fif_subdir=False, verbose=False)
    trigger = TriggerSoftware(recorder=recorder, verbose=False)
    trigger.signal(1)
    time.sleep(duration)
    trigger.close()
    recorder.stop()

#%%
#
# Alternative paradigm function which prints every seconds to keep track of the
# progression. The `time.sleep()` is replaced with a while loop.

def resting_state_with_verbose(directory, stream_name, duration):
    """
    Function called in __main__.
    """
    recorder = StreamRecorder(directory)
    recorder.start(fif_subdir=False, verbose=False)
    trigger = TriggerSoftware(recorder=recorder, verbose=True)
    timer = Timer()
    previous_time_printed = 0
    trigger.signal(1)
    timer.reset()
    while timer.sec() <= duration:
        if previous_time_printed+1 <= timer.sec():
            previous_time_printed += 1
            print (datetime.timedelta(seconds=previous_time_printed))
    trigger.close()
    recorder.stop()

#%%
#
# Call in `__main__` the function. The StreamRecorder starts a new process,
# which can not be done outside `__main__` on Windows.
# See: https://docs.python.org/2/library/multiprocessing.html#windows

if __name__ == '__main__':
    resting_state(directory, stream_name, duration)
    # resting_state_with_verbose(directory, stream_name, duration)

#%%
#
# Stop the mock LSL stream.

if __name__ == '__main__':
    player.stop()
