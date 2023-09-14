"""
Stream information
==================

.. include:: ./../../links.inc
"""

# %%
# bla

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
