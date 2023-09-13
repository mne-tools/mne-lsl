"""
Player
======

.. include:: ./../../links.inc

During the development of a project, it's very helpful to test on a mock LSL stream
replicating our experimental condition. The :class`~bsl.Player` can create a mock LSL
stream from any `MNE <mne stable_>`_ readable file.

.. note::

    For now, the mock capabilities are restricted to streams with a continuous sampling
    rate. Streams with an irregular sampling rate corresponding to event streams are not
    yet supported.
"""

# %%
# bla
