"""
====================================
StreamPlayer: simulate an LSL stream
====================================

Testing designs for online paradigm can be difficult. Access to hardware
measuring real-time brain signals can be limited and time-consuming. With a
`~bsl.StreamPlayer`, a fake data stream can be created and used to test code
and experiment designs.
"""

#%%

# Authors: Mathieu Scheltienne <mathieu.scheltienne@fcbg.ch>
#
# License: LGPL-2.1

# sphinx_gallery_thumbnail_path = '_static/stream_player/stream_player_cli.gif'

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
# directory, in the folder ``bsl_data``.

#%%

import time

from bsl import StreamPlayer, datasets
from bsl.lsl import resolve_streams
from bsl.triggers import TriggerDef

#%%
# Starting a StreamPlayer
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# A `~bsl.StreamPlayer` requires at least 2 arguments:
#
# - ``stream_name``, indicating a the name of the stream on the LSL network.
# - ``fif_file``, path to a valid `~mne.io.Raw` fif file.

stream_name = 'StreamPlayer'
fif_file = datasets.eeg_resting_state.data_path()
print (fif_file)

#%%
# Instance
# """"""""
#
# To create an LSL stream, create a `~bsl.StreamPlayer` and use the
# `~bsl.StreamPlayer.start` method.
#
# .. note::
#
#     By default, the `~bsl.StreamPlayer.start` method is blocking and will
#     wait for the streaming to start on the network. This behavior can be
#     changed with the ``blocking`` argument.

player = StreamPlayer(stream_name, fif_file)
player.start()
print (player)

#%%
#
# To verify if the stream is accessible on the network, use directly ``pylsl``:

streams = [stream.name for stream in resolve_streams()]
print (streams)

#%%
#
# To stop the streaming, use the `~bsl.StreamPlayer.stop` method.

player.stop()
print (player)

#%%
# Context manager
# """""""""""""""
#
# A `~bsl.StreamPlayer` can also be used as a context manager with a ``with``
# statement. The context manager takes care of starting and stopping the LSL
# stream.

with StreamPlayer(stream_name, fif_file):
    streams = [stream.name for stream in resolve_streams()]
print (streams)

#%%
# CLI
# """
#
# Finally, a `~bsl.StreamPlayer` can be called from the terminal with a command
# line. This is the recommended way of starting a `~bsl.StreamPlayer`.
# Example assuming the current working directory is ``bsl_data``:
#
# .. code-block:: console
#
#     $ bsl_stream_player StreamPlayer eeg_sample\resting_state-raw.fif
#
# Hitting ``ENTER`` will stop the `~bsl.StreamPlayer`.
#
# .. image:: ../_static/stream_player/stream_player_cli.gif
#    :alt: StreamPlayer
#    :align: center

#%%
# Additional arguments
# ^^^^^^^^^^^^^^^^^^^^
#
# A `~bsl.StreamPlayer` has 4 optional arguments:
#
# - ``repeat``, indicating the number of time the data in the file is repeated.
# - ``trigger_def``, either the path to a :class:`.TriggerDef` definition file
#   or a :class:`.TriggerDef` instance, improving the logging of events found
#   in the `~mne.io.Raw` fif file.
# - ``chunk_size``, indicating the number of samples push at once on the LSL
#   outlet.
# - ``high_resolution``, indicating if `~time.sleep` or `~time.perf_counter` is
#   used to wait between 2 push on the LSL outlet.

#%%
# repeat
# """"""
#
# ``repeat`` is set by default to ``+inf``, returning to the beginning of the
# data in the `~mne.io.Raw` fif file each time the entire file has been 2
# streamed. To limit the number of replay, an `int` can be passed.
#
# .. note::
#
#     `~bsl.datasets.eeg_resting_state_short` is similar to
#     `~bsl.datasets.eeg_resting_state` but last 2 seconds instead of 40
#     seconds.

fif_file = datasets.eeg_resting_state_short.data_path()
player = StreamPlayer(stream_name, fif_file, repeat=1)
player.start()
print (player)

#%%
#
# The dataset is streamed only once. A call to the `~bsl.StreamPlayer.stop`
# method is not necessary.

time.sleep(2)  # duration of this dataset.
print (player)

#%%
# trigger_def
# """""""""""
#
# :class:`.TriggerDef` can be used to assign a user-readable string to an event
# id. Providing a valid :class:`.TriggerDef` to a `~bsl.StreamPlayer` improves
# the logging of events found on the ``TRIGGER`` channel.
#
# .. note::
#
#     `~bsl.datasets.eeg_auditory_stimuli` contains an alternation of rest
#     events (1) lasting 1 second and of auditory stimuli events (4) lasting
#     0.8 second.

fif_file = datasets.eeg_auditory_stimuli.data_path()
player = StreamPlayer(stream_name, fif_file)
player.start()
print (player)

# wait a bit to get some events logged
time.sleep(4)

# stop
player.stop()
print (player)

#%%
#
# By default, the logging of events uses the ID with ``Events: ID``. If a
# :class:`.TriggerDef` is provided, the logging message will include the
# corresponding event name if it exists with ``Events: ID (NAME)``.

tdef = TriggerDef()
tdef.add('rest', 1)

player = StreamPlayer(stream_name, fif_file, trigger_def=tdef)
player.start()
print (player)

# wait a bit to get some events logged
time.sleep(4)

# stop
player.stop()
print (player)

#%%
#
# .. note::
#
#     A path to a valid ``.ini`` trigger definition file can be passed instead
#     of a :class:`TriggerDef` instance. The file is read with
#     `configparser` and has to be structured as follows:
#
#     .. code-block:: python
#
#         [events]
#         event_str_1 = event_id_1   # comment
#         event_str_2 = event_id_2   # comment
#
#     Example:
#
#     .. code-block:: python
#
#         [events]
#         rest = 1
#         stim = 2

#%%
# chunk_size
# """"""""""
#
# ``chunk_size`` defines how many samples are pushed at once on the LSL oulet
# each time the `~bsl.StreamPlayer` sends data. The default ``16`` should work
# most of the time. A warning is emitted if the value is different from the
# usual ``16`` or ``32``.

#%%
# high_resolution
# """""""""""""""
#
# Between 2 push of samples on the LSL outlet, the `~bsl.StreamPlayer` waits.
# This sleep duration can be achieved either with `~time.sleep` or with
# `~time.perf_counter`. The second is more precise, but also uses more CPU.
