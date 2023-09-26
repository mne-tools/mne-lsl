"""
Stream information
==================

.. include:: ./../../links.inc

A :class:`~bsl.Stream` will automatically attempt to interpret the channel names, types
and units during the connection with :meth:`bsl.Stream.connect`. However, by definition,
an LSL stream does not require any of those information to be present.
Moreover, the channel type and unit are not standardize, and may be define with
different nomenclature depending on the system and the application emitting the LSL
stream. For instance, an EEG channel might be denoted by the type ``'eeg'`` or
``'electroencephalography'``, or something else entirely.

If ``BSL`` is not able to interpret the information in a stream description, it will
default to:

* numbers instead of channel names if it failed to load the channel names, ``'0'```,
  ``'1'``, ... as :func:`mne.create_info` does when the argument ``ch_names`` is
  provided as a number of channels.
* The stream type (if interpretable) or ``'misc'`` otherwise if it failed to load the
  individual channel types.
* SI units (factor 0) if it failed to load the individual channel units.

The stream and channel type supported correspond to the MNE-supported channel types.
"""

# %%
# Inspecting a stream info
# ------------------------
#
# A :class:`~bsl.Stream` measurement information can be inspected with similar methods
# to a :class:`~mne.io.Raw` object: :py:attr:`bsl.Stream.info`,
# :py:attr:`bsl.Stream.ch_names`, :meth:`bsl.Stream.get_channel_types`,
# :meth:`bsl.Stream.get_channel_units`.
#
# .. note::
#
#     For this tutorial purposes, a mock LSL stream is created using a
#     :class:`~bsl.Player`. See :ref:`sphx_glr_generated_tutorials_10_player.py` for
#     additional information on mock LSL streams.

from bsl import Player, Stream
from bsl.datasets import sample

fname = sample.data_path() / "sample-ant-aux-raw.fif"
player = Player(fname)
player.start()
stream = Stream(bufsize=5)  # 5 seconds of buffer
stream.connect()
stream.info

# %%
# :py:attr:`bsl.Stream.ch_names` and :meth:`bsl.Stream.get_channel_types` behave like
# their `MNE <mne stable_>`_ counterpart, but :meth:`bsl.Stream.get_channel_units` is
# unique to ``BSL``. In `MNE <mne stable_>`_, recordings are expected to be provided in
# SI units, and it is up to the end-user to ensure that the underlying data array is
# abiding.
#
# However, many system do not stream data in SI units. For instance, most EEG amplifiers
# stream data in microvolts. ``BSL`` implements a 'units' API to handle the difference
# in units between 2 stream of similar sources, e.g. between an EEG stream from a first
# amplifier in microvolts and an EEG stream from a second amplifier in nanovolts.

# look at the 3 channels with the type 'eeg'
ch_types = stream.get_channel_types(picks="eeg")
ch_units = stream.get_channel_units(picks="eeg")
for ch_name, ch_type, ch_unit in zip(stream.ch_names, ch_types, ch_units):
    print(f"Channel '{ch_name}' of type '{ch_type}' has the unit '{ch_unit}'.")

# %%
# In our case, the 3 selected channels have the unit
# ``((107 (FIFF_UNIT_V), 0 (FIFF_UNITM_NONE))``. This format contains 2 elements:
#
# * The first element, ``107 (FIFF_UNIT_V)``, gives the unit type/family. In this case,
#   ``V`` means that the unit type is ``Volts``. Each sensor type is associated to a
#   different unit type, thus to change the first element the sensor type must be set
#   with :meth:`bsl.Stream.set_channel_types`.
# * The second element, ``0 (FIFF_UNITM_NONE))``, gives the unit scale (Giga, Kilo,
#   micro, ...) in the form of the power of 10 multiplication factor. In this case,
#   ``0`` means ``e0``, i.e. ``10**0``.
#
# Thus, the unit stored is ``Volts``, corresponding to the SI unit for
# electrophysiological channels.
#
# Correct a stream info
# ---------------------
#
# If a :py:attr:`bsl.Stream.info` does not contain the correct attributes, it should be
# corrected similarly as for a :class:`~mne.io.Raw` object. In this case:
#
# * the channel ``AUX1`` is a vertical EOG channel.
# * the channel ``AUX2`` is an ECG channel.
# * the channel ``AUX3`` is an horizontal EOG channel.

stream.rename_channels({"AUX1": "vEOG", "AUX2": "ECG", "AUX3": "hEOG"})
stream.set_channel_types({"vEOG": "eog", "hEOG": "eog", "ECG": "ecg"})
stream.info

# %%
# TODO: section about setting the channel units

# %%
# Free resources
# --------------
# When you are done with a :class:`~bsl.Player` or :class:`~bsl.Stream`, don't forget
# to free the resources they both use to continuously mock an LSL stream or receive new
# data from an LSL stream.

stream.disconnect()
player.stop()
