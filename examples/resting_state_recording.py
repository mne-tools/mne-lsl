"""
=======================================
StreamRecorder: resting-state recording
=======================================

A resting-state recording is a simple offline recording during which the brain
activity of a subject is measured in the absence of any stimulus or task. A
resting-state recording can be designed with a `~bsl.StreamRecorder`.
"""

#%%

# Authors: Mathieu Scheltienne <mathieu.scheltienne@gmail.com>
#
# License: LGPL-2.1

# sphinx_gallery_thumbnail_path = '_static/stream_recorder/stream_recorder_cli.gif'

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

from bsl import StreamPlayer, StreamRecorder, datasets
from bsl.triggers.software import SoftwareTrigger

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
#
# For this example, the folder ``bsl_data/examples`` located in the user home
# directory will be used to stored recorded files. To ensure its existence,
# `os.makedirs` is used.

record_dir = Path('~/bsl_data/examples').expanduser()
os.makedirs(record_dir, exist_ok=True)
print (record_dir)

#%%
#
# For this simple offline recording, the goal is to start a
# `~bsl.StreamRecorder`, send an event on a trigger to mark the beginning of
# the resting-state recording, wait for a defined duration, and stop the
# recording.
#
# By default, a `~bsl.StreamRecorder` does not require any argument. The
# current working directory is used to record data from all available streams
# in files named based on the date/time timestamp at which the recorder is
# started.
#
# To record only a subset of the available streams with a specific file name
# and in a specific directory, the arguments ``record_dir``, ``fname`` and
# ``stream_name`` must be provided.
#
# For this example, the directory used to store recordings is
# ``bsl_data/examples`` and the file name will start with
# ``example-resting-state``.
#
# .. note::
#
#     By default, the `~bsl.StreamRecorder.start` method is blocking and will
#     wait for the recording to start. This behavior can be changed with the
#     ``blocking`` argument.

recorder = StreamRecorder(record_dir, fname='example-resting-state')
recorder.start()
print (recorder)

#%%
#
# Now that a `~bsl.StreamRecorder` is started and is acquiring data, a trigger
# to mark the beginning of the segment of interest is created. For this
# example, a `~bsl.triggers.software.SoftwareTrigger` is used, but this example
# would be equally valid with a different type of trigger.
#
# .. note::
#
#     `~bsl.triggers.software.SoftwareTrigger` must be created after a
#     `~bsl.StreamRecorder` is started and closed/deleted before a
#     `~bsl.StreamRecorder` is stopped.
#
#     .. code-block:: python
#
#         recorder = StreamRecorder()
#         recorder.start()
#         trigger = SoftwareTrigger(recorder)
#         # do stuff
#         trigger.close() # OR >>> del trigger
#         recorder.stop()
#
#     All triggers do not need an active `~bsl.StreamRecorder` to be created.

trigger = SoftwareTrigger(recorder, verbose=True)

#%%
#
# To mark the beginning of the segment of interest in the recording, a signal
# is sent on the trigger. For this example, the event value (1) is used.

trigger.signal(1)

#%%
#
# Finally, after the appropriate duration, the recording is interrupted.
#
# .. note::
#
#     `~bsl.triggers.software.SoftwareTrigger` must be closed or deleted before
#     the recorder is stopped. All triggers do not need to be closed or deleted
#     before the recorder is stopped.

time.sleep(2)  # 2 seconds duration
trigger.close()
recorder.stop()
print (recorder)

#%%
#
# A `~bsl.StreamRecorder` records data in ``.pcl`` format. This file can be
# open with `pickle.load`, and is automatically converted to a `~mne.io.Raw`
# FIF file in a subdirectory ``fif``. The recorded files name syntax is:
#
# - If ``fname`` is not provided: ``[date/time timestamp]-[stream]-raw.fif``
# - If ``fname`` is provided: ``[fname]-[stream]-raw.fif``
#
# Where ``stream`` is the name of the recorded LSL stream. Thus, one file is
# created for each stream being recorded.

fname = record_dir / 'fif' / 'example-resting-state-StreamPlayer-raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
print (raw)
events = mne.find_events(raw, stim_channel='TRIGGER')
print (events)

#%%
#
# As for the `~bsl.StreamPlayer`, the `~bsl.StreamRecorder` can be used as a
# context manager. The context manager takes care of starting and stopping the
# recording.

with StreamRecorder(record_dir):
    time.sleep(1)

#%%
#
# As for the `~bsl.StreamPlayer`, the `~bsl.StreamRecorder` can be started via
# command-line when a LSL stream is accessible on the network.
#
# Example assuming:
#
# - the current working directory is ``bsl_data`` in the user home directory
# - the stream to connect to is named ``MyStream``
# - the recorded file naming scheme is ``test-[stream]-raw.fif``, i.e.
#   ``test-MyStream-raw.fif``
#
# .. code-block:: console
#
#     $ bsl_stream_recorder -d examples -f test -s MyStream
#
# .. image:: ../_static/stream_recorder/stream_recorder_cli.gif
#    :alt: StreamRecorder
#    :align: center

#%%
# | Stop the mock LSL stream used in this example.

player.stop()
