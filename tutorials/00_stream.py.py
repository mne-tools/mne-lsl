"""
Stream
======

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
# To begin this tutorial, we will create a mock LSL stream with a :class:`~bsl.Player`
# and we will connect to this stream with a :class:`~bsl.Stream`.

from bsl import Player, Stream
from bsl.datasets import sample
from mne import pick_types

fname = sample.data_path() / "sample-ant-raw.fif"
player = Player(fname)
player.start()
stream = Stream(bufsize=5)  # 5 seconds of buffer
stream.connect()

# sphinx_gallery_thumbnail_path = '_static/tutorials/circular-buffer-light.png'

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
# and :meth:`bsl.Stream.set_channel_units`.
#
# For instance, let's have a closer look at the EEG channel units with
# :meth:`bsl.Stream.get_channel_units`.

units = stream.get_channel_units(picks="eeg")
print (set(units))  # remove duplicates

#%%
# In our case, all EEG channels have the unit
# ``((107 (FIFF_UNIT_V), 0 (FIFF_UNITM_NONE))``. This format only looks complicated:
#
# * The first element, ``107 (FIFF_UNIT_V)``, gives the unit type/family. In this case,
#   ``V`` means that the unit type is ``Volts``.
# * The second element, ``0 (FIFF_UNITM_NONE))``, gives the unit scale (Giga, Kilo,
#   micro, ...) in the form of the power of 10 multiplication factor. In this case,
#   ``0`` means ``e0``, i.e. ``10**0``.
#
# Thus, the unit is ``Volts``, corresponding to the SI unit for EEG channels.
#
# But most amplifier streams are in microvolts, thus if the unit read by the
# :class:`~bsl.Stream` in the ``.info`` attribute does not correspond to the reality,
# you can change it with :meth:`bsl.Stream.set_channel_types`.

mapping = {stream.ch_names[k]: "microvolts" for k in pick_types(stream.info, eeg=True)}
stream.set_channel_units(mapping)
units = stream.get_channel_units(picks="eeg")
print (set(units))  # remove duplicates

#%%
# Note that the unit type did not change but the multiplication factor is now set to
# ``-6 (FIFF_UNITM_MU)`` corresponding to ``μV``.
#
# .. note::
#
#     The unit can be provided as the power of 10 multiplication factor, e.g. ``-6`` for
#     micro- or as readable string for known channel types and units. ``microvolts`` or
#     ``uv`` is common enough to be correctly interpreted by ``BSL``. If you use a unit
#     which is not understood by ``BSL`` and would like to add it to the known units,
#     please open an issue on GitHub.

# %%
# Free resources
# --------------
# When you are done with a :class:`~bsl.Player` or :class:`~bsl.Stream`, don't forget
# to free the resources they both use to continuously mock an LSL stream or receive new
# data from an LSL stream.

stream.disconnect()
player.stop()
