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

Both types can be managed through a :class:`bsl.Stream` object, which represents a
single LSL stream with its buffer containing the current and past samples. The buffer
size is specified at instantiation through the ``bufsize`` argument.
"""

# %%
# Internal ringbuffer
# -------------------
#
# Once the :class:`~bsl.Stream` object is connected to an LSL Stream, it automatically
# updates an internal ringbuffer with newly available samples. A ringbuffer, also called
# circular buffer, is a data structure that uses a single fixed-size buffer as if it
# were connected and to end.
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
# With a `~bsl.Stream`, the pointers are hidden and the head pointer is always updated
# to the last received sample.

# %%
# Connect to a Stream
# -------------------
#
# Connecting to an LSL Stream is a 2 step operation. First, create a
# :class:`~bsl.Stream` with the desired buffer size and the desired stream attributes,
# ``name``, ``stype``, ``source_id``. Second, connect to the stream which matches the
# requested stream attributes with :meth:`bsl.Stream.connect`.
#
# .. note::
#
#     For this tutorial purposes, a mock LSL stream is created using a
#     :class:`~bsl.Player`. See :ref:`sphx_glr_generated_tutorials_10_player.py` for
#     additional information on mock LSL streams.

from bsl import Player, Stream
from bsl.datasets import sample

fname = sample.data_path() / "sample-ant-raw.fif"
player = Player(fname)
player.start()
stream = Stream(bufsize=5)  # 5 seconds of buffer
stream.connect()

#%%
# Stream information
# ------------------
#
# Similar to a :class:`~mne.io.Raw` recording and to most `MNE <mne stable_>`_ objects,
# a :class:`~bsl.Stream` has an ``.info`` attribute containing the channel names, types
# and units.

stream.info

#%%
# Depending on the LSL Stream source, the `~bsl.Stream` may or may not be able to
# correctly read the channel names, types and units.
#
# * If the channel names are not readable or present, numbers will be used.
# * If the channel types are not readable or present, the stream type or ``'misc'`` will
#   be used.
# * If the channel units are not readable or present, SI units will be used.
#
# Once connected to a Stream, you can change the channel names, types and units to your
# liking with :meth:`bsl.Stream.rename_channels`, :meth:`bsl.Stream.set_channel_types`
# and :meth:`bsl.Stream.set_channel_units`. See
# :ref:`sphx_glr_generated_tutorials_20_stream_meas_info.py` for additional information.

# %%
# Free resources
# --------------
# When you are done with a :class:`~bsl.Player` or :class:`~bsl.Stream`, don't forget
# to free the resources they both use to continuously mock an LSL stream or receive new
# data from an LSL stream.

stream.disconnect()
player.stop()
