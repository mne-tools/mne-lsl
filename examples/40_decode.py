"""
Decoding real-time data
=======================

.. include:: ./../../links.inc

This example demonstrates how to decode real-time data using `MNE-Python <mne stable_>`_
and `Scikit-learn <sklearn stable_>`_. We will stream the ``sample_audvis_raw.fif``
file from MNE's sample dataset with a :class:`~mne_lsl.player.PlayerLSL`, process the
signal through a :class:`~mne_lsl.stream.StreamLSL`, and decode the epochs created with
:class:`~mne_lsl.stream.EpochsStream`.
"""

import uuid

from mne.io import read_raw_fif

from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import EpochsStream, StreamLSL

fname = sample.data_path() / "mne-sample" / "sample_audvis_raw.fif"
raw = read_raw_fif(fname, preload=False).pick(("meg", "stim")).load_data()
source_id = uuid.uuid4().hex
player = PlayerLSL(raw, chunk_size=200, name="real-time-decoding", source_id=source_id)
player.start()
player.info

# %%
# Signal processing
# -----------------
#
# We will apply minimal signal processing to the data. First, only the gradiometers will
# be used for decoding, thus other channels are removed. Then we mark bad channels and
# applying a low-pass filter at 40 Hz.

stream = StreamLSL(bufsize=5, name="real-time-decoding", source_id=source_id)
stream.connect(acquisition_delay=0.1, processing_flags="all")
stream.info["bads"] = ["MEG 2443"]
stream.pick(("grad", "stim")).filter(None, 40, picks="grad")
stream.info

# %%
# Epoch the signal
# ----------------
#
# Next, we will create epochs around the event ``1`` (audio left) and ``3`` (visual
# left).

epochs = EpochsStream(
    stream,
    bufsize=10,
    event_id=dict(audio_left=1, visual_left=3),
    event_channels="STI 014",
    tmin=-0.2,
    tmax=0.5,
    baseline=(None, 0),
    reject=dict(grad=4000e-13),  # unit: T / m (gradiometers)
).connect(acquisition_delay=0.1)

# %%
# Free resources
# --------------
#
# When you are done with a :class:`~mne_lsl.player.PlayerLSL`,
# :class:`~mne_lsl.stream.StreamLSL` ir :class:`~mne_lsl.stream.EpochsStream`, don't
# forget to free the resources they use to continuously mock an LSL stream or receive
# new data from an LSL stream.

epochs.disconnect()

# %%

stream.disconnect()

# %%

player.stop()
