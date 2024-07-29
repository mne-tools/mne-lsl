"""
Real-time evoked responses
==========================

With a :class:`~mne_lsl.stream.EpochsStream`, we can build a real-time evoked response
vizualization. This is useful to monitor the brain activity in real-time.
"""

import numpy as np
from matplotlib import pyplot as plt
from mne import EvokedArray, combine_evoked
from mne.datasets import sample
from mne.io import read_raw_fif

from mne_lsl.player import PlayerLSL
from mne_lsl.stream import EpochsStream, StreamLSL

# dataset used in the example
data_path = sample.data_path()
fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
raw = read_raw_fif(fname, preload=False).pick(("meg", "stim")).load_data()

# %%
# First, we create a mock stream with :class:`mne_lsl.player.PlayerLSL` from the sample
# dataset and connect a :class:`~mne_lsl.stream.StreamLSL` to it. Then, we attach a
# :class:`~mne_lsl.stream.EpochsStream` object to create epochs from the LSL stream.
# The epochs will be created around the event ID ``1`` from the ``'STI 014'`` channel.

with PlayerLSL(raw, chunk_size=200, name="real-time-evoked-example"):
    stream = StreamLSL(bufsize=4, name="real-time-evoked-example")
    stream.connect(acquisition_delay=0.1, processing_flags="all")
    epochs = EpochsStream(
        stream,
        bufsize=20,
        event_id=1,
        event_channels="STI 014",
        tmin=-0.2,
        tmax=0.5,
        baseline=(None, 0),
        picks="grad",
    )
    epochs.connect(acquisition_delay=0.1)

    # start looking for epochs
    n = 0  # number of epochs
    evoked = None
    while n <= 20:
        if epochs.n_new_epochs == 0:
            continue  # nothing new to do
        # get data and create evoked array
        data = epochs.get_data(only_new=True)
        new_evoked = EvokedArray(
            np.average(data, axis=0), epochs.info, nave=data.shape[0]
        )
        evoked = (
            new_evoked
            if evoked is None
            else combine_evoked([evoked, new_evoked], weights="nave")
        )
        plt.clf()  # clear canvas
        evoked.plot(axes=plt.gca(), time_unit="s")  # plot on current figure
        plt.pause(0.05)
