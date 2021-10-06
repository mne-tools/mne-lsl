"""
===========================================
Simulate an LSL stream with a Stream Player
===========================================
The stream player creates a fake LSL stream to test code, paradigm which needs
to connect to a stream.
"""

# Authors: Mathieu Scheltienne <mathieu.scheltienne@gmail.com>
#
# License: LGPL-2.1

#%%

import time

from bsl import StreamPlayer, datasets

#%%
#
# Define a name for the LSL stream.
# Define the path to a .fif file recorded with the Stream Recorder.
# As currently, the only data stream supported is EEG, all channels except the
# trigger channel are multiplied by 1e6 to convert from Volts to uVolts.

stream_name = 'MyStreamPlayer'
sample_data_raw_file = datasets.eeg_resting_state.data_path()
print (sample_data_raw_file)

#%%
#
# Define a stream player and start streaming during a fix duration.
# Call in `__main__` because the Stream Player starts a new process, which can
# not be done outside `__main__` on Windows.
# See: https://docs.python.org/2/library/multiprocessing.html#windows

if __name__ == '__main__':
    player = StreamPlayer(stream_name, sample_data_raw_file)
    player.start()
    time.sleep(3)  # fix 3 seconds duration.
    player.stop()

#%%
#
# A stream player can be called directly from the command line with:
# `bsl_stream_player MyStreamPlayer "path to -raw.fif"`.
