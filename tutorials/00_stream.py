"""
Introduction to the Stream API
==============================

.. include:: ./../../links.inc

An LSL stream can be consider as a continuous recording, with an unknown length and with
only access to the current and past samples. LSL streams can be separate in 2
categories:

* Streams with a **regular** sampling rate, which can be considered as a
  :class:`~mne.io.Raw` continuous recording.
* Streams with an **irregular** sampling rate, which can be considered as spontaneous
  events.

Both types can be managed through a ``Stream`` object, which represents a
single LSL stream with its buffer containing the current and past samples. The buffer
size is specified at instantiation through the ``bufsize`` argument.
"""

# %%
# Internal ringbuffer
# -------------------
#
# Once the :class:`~bsl.stream.StreamLSL` object is connected to an LSL Stream, it
# automatically updates an internal ringbuffer with newly available samples. A
# ringbuffer, also called circular buffer, is a data structure that uses a single
# fixed-size buffer as if it were connected and to end.
#
# .. image:: ../../_static/tutorials/circular-buffer-light.png
#     :align: center
#     :class: only-light
#
# .. image:: ../../_static/tutorials/circular-buffer-dark.png
#     :align: center
#     :class: only-dark
#
# Typically, a ring buffer has 2 pointers:
#
# * The "head" pointer, also called "start" or "read", which corresponds to the next
#   data block to read.
# * The "tail" pointer, also called "end" or "write", which corresponds to the next
#   data block that will be overwritten with new data.
#
# With a `~bsl.stream.StreamLSL`, the pointers are hidden and the head pointer is always
# updated to the last received sample.

# %%
# Connect to a Stream
# -------------------
#
# Connecting to an LSL Stream is a 2 step operation. First, create a
# :class:`~bsl.stream.StreamLSL` with the desired buffer size and the desired stream
# attributes, ``name``, ``stype``, ``source_id``. Second, connect to the stream which
# matches the requested stream attributes with :meth:`bsl.stream.StreamLSL.connect`.
#
# .. note::
#
#     For this tutorial purposes, a mock LSL stream is created using a
#     :class:`~bsl.player.PlayerLSL`. See
#     :ref:`sphx_glr_generated_tutorials_10_player.py` for additional information on
#     mock LSL streams.

import time

from matplotlib import pyplot as plt

from bsl.datasets import sample
from bsl.lsl import local_clock
from bsl.player import PlayerLSL as Player
from bsl.stream import StreamLSL as Stream

fname = sample.data_path() / "sample-ant-raw.fif"
player = Player(fname)
player.start()
stream = Stream(bufsize=5)  # 5 seconds of buffer
stream.connect()

# %%
# Stream information
# ------------------
#
# Similar to a :class:`~mne.io.Raw` recording and to most `MNE <mne stable_>`_ objects,
# a :class:`~bsl.stream.StreamLSL` has an ``.info`` attribute containing the channel
# names, types and units.

stream.info

# %%
# Depending on the LSL Stream source, the `~bsl.stream.StreamLSL` may or may not be able
# to correctly read the channel names, types and units.
#
# * If the channel names are not readable or present, numbers will be used.
# * If the channel types are not readable or present, the stream type or ``'misc'`` will
#   be used.
# * If the channel units are not readable or present, SI units will be used.
#
# Once connected to a Stream, you can change the channel names, types and units to your
# liking with :meth:`bsl.stream.StreamLSL.rename_channels`,
# :meth:`bsl.stream.StreamLSL.set_channel_types` and
# :meth:`bsl.stream.StreamLSL.set_channel_units`. See
# :ref:`sphx_glr_generated_tutorials_20_stream_meas_info.py` for additional information.

# %%
# Channel selection
# -----------------
#
# Channels can be selected with :meth:`bsl.stream.StreamLSL.pick` or with
# :meth:`bsl.stream.StreamLSL.drop_channels`. Selection is definitive, it is not
# possible to restore channels removed until the :class:`~bsl.stream.StreamLSL` is
# disconnected and reconnected to its source.

stream.pick(["Fz", "Cz", "Oz"])
stream.info

# %%
# Query the buffer
# ----------------
#
# The ringbuffer can be queried for the last ``N`` samples with
# :meth:`bsl.stream.StreamLSL.get_data`. The argument ``winsize`` controls the amount of
# samples returned, and the property :py:attr:`bsl.stream.StreamLSL.n_new_samples`
# contains the amount of new samples buffered between 2 queries.
#
# .. note::
#
#     If the number of new samples between 2 queries is superior to the number of
#     samples that can be hold in the buffer :py:attr:`bsl.stream.StreamLSL.n_buffer`, the
#     buffer is overwritten with some samples "lost" or discarded without any prior
#     notice or error raised.

print(f"Number of new samples: {stream.n_new_samples}")
data, ts = stream.get_data()
time.sleep(0.5)
print(f"Number of new samples: {stream.n_new_samples}")

# %%
# :meth:`bsl.stream.StreamLSL.get_data` returns 2 variables, ``data`` which contains the
# ``(n_channels, n_samples)`` data array and ``ts`` (or ``timestamps``) which contains
# the ``(n_samples,)`` timestamp array, in LSL time.
#
# .. note::
#
#     LSL timestamps are not always regular. They can be jittered depending on the source
#     and on the delay between the source and the client. Processing flags can be
#     provided to improve the timestamp precision when connecting to a stream with
#     :meth:`bsl.stream.StreamLSL.connect`. See
#     :ref:`sphx_glr_generated_tutorials_30_timestamps.py` for additional information.

t0 = local_clock()
f, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True)
for _ in range(3):
    # figure how many new samples are available, in seconds
    winsize = stream.n_new_samples / stream.info["sfreq"]
    # retrieve and plot data
    data, ts = stream.get_data(winsize)
    for k, data_channel in enumerate(data):
        ax[k].plot(ts - t0, data_channel)
    time.sleep(0.5)
for k, ch in enumerate(stream.ch_names):
    ax[k].set_title(f"EEG {ch}")
ax[-1].set_xlabel("Timestamp (LSL time)")
plt.show()

# %%
# In the previous figure, the timestamps are corrected by ``t0``, which correspond to
# the time at which the first loop was executed. Note that the samples in blue span
# negative time values. Indeed, a 0.5 second sleep was added in the previous code cell
# after the last :meth:`bsl.stream.StreamLSL.get_data` call. Thus, ``t0`` is created 0.5
# seconds after the last reset of :py:attr:`bsl.stream.StreamLSL.n_new_samples` and the
# samples pulled with the first :meth:`bsl.stream.StreamLSL.get_data` correspond to past
# samples.
#
# Note also the varying number of samples in each of the 3 data query separated by
# 0.5 seconds. When connecting to a Stream with :meth:`bsl.stream.StreamLSL.connect`, an
# ``acquisition_delay`` is defined. It corresponds to the delay between 2 updates of the
# ringbuffer, by default 200 ms. Thus, with a 500 ms sleep in this example, the number
# of samples updated in the ringbuffer will vary every 2 iterations.

# %%
# Apply processing to the buffer
# ------------------------------
#
# TODO: add_reference_channels, set_eeg_reference, filter

# %%
# Record a stream
# ---------------
#
# TODO

# %%
# Free resources
# --------------
# When you are done with a :class:`~bsl.player.PlayerLSL` or
# :class:`~bsl.stream.StreamLSL`, don't forget to free the resources they both use to
# continuously mock an LSL stream or receive new data from an LSL stream.

stream.disconnect()
player.stop()
