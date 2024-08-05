"""
Epoching a Stream in real-time
==============================

.. include:: ./../../links.inc

The :class:`~mne_lsl.stream.EpochsStream` object can be used similarly to
:class:`mne.Epochs` to create epochs from a continuous stream of samples around events
of interest.

.. note::

    The :class:`~mne_lsl.stream.EpochsStream` object is designed to work with
    any ``Stream`` object. At the time of writing, only
    :class:`~mne_lsl.stream.StreamLSL` is available, but any object inheriting from the
    abstract :class:`~mne_lsl.stream.BaseStream` object should work.
"""
