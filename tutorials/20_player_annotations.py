"""
.. _tut-player-annotations:

Player with annotations
=======================

.. include:: ./../../links.inc

:class:`~mne.Annotations` from a :class:`~mne.io.Raw` object can be streamed as an event
stream by :class:`~mne_lsl.player.PlayerLSL`. The stream will be irregularly sampled,
numerical, and of type ``'annotations'``.

A :class:`~mne.Annotations` contain 3 information:

- the onset of the annotation
- the duration of the annotation
- the description of the annotation

To stream all 3 information, it's duration-hod encoded along the channels. For instance,
consider a :class:`~mne.io.Raw` object with 3 different :class:`~mne.Annotations`
description: ``'event1'``, ``'event2'``, and ``'event3'``. The event stream will have 3
channels, each corresponding to one of the 3 descriptions. When an annotation is
streamed, it's duration is encoded as the value on its channel while the other channels
remain to zero.
"""

import uuid

import numpy as np
from matplotlib import pyplot as plt
from mne import Annotations, create_info
from mne.io import RawArray
from mne.viz import set_browser_backend

from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL


annotations = Annotations(
    onset=[1, 2, 3],
    duration=[0.1, 0.2, 0.3],
    description=["event1", "event2", "event3"],
)
annotations

# %%
# With the 3 annotations above, the event stream will stream the following samples:
#
# - at time ``1``, the annotation ``'event1'`` is pushed. The sample push is
#   ``array([[0.1, 0, 0]])``, of shape (1, 3).
# - at time ``2``, the annotation ``'event2'`` is pushed. The sample push is
#   ``array([[0, 0.2, 0]])``, of shape (1, 3).
# - at time ``3``, the annotation ``'event3'`` is pushed. The sample push is
#   ``array([[0, 0, 0.3]])``, of shape (1, 3).
#
# If more than one annotations are present in the chunk currently pushed, then a chunk
# is pushed. For instance, if the annotations at time ``2`` and ``3`` are pushed at the
# same time, then the chunk push is ``array([[0., 0.2., 0.], [0., 0., 0.3]])``, of shape
# (2, 3).
#
# %%
# Example on mock signal
# ----------------------
#
# Let's create a mock :class:`~mne.io.Raw` object with annotations and stream both the
# signal and the annotations.

data = np.zeros((1, 1000))  # 1 channel, 1000 samples
data[0, 100:200] = 1
data[0, 500:700] = 2
info = create_info(["signal"], 1000, "misc")
raw = RawArray(data, info)
annotations = Annotations(onset=[0.1, 0.5], duration=[0.1, 0.2], description=["1", "2"])
raw.set_annotations(annotations)
set_browser_backend("matplotlib")  # easier to plot with matplotlib in a documentation
raw.plot(scalings=dict(misc=2), show_scrollbars=False, show_scalebars=False)
plt.show()

# %%
# Now that we have the :class:`~mne.io.Raw` object, we can stream it with a
# :class:`~mne_lsl.player.PlayerLSL` object.
#
# .. note::
#
#     Note that forcing ``annotations=True`` is not necessary since the
#     :class:`~mne_lsl.player.PlayerLSL` will automatically stream annotations if they
#     are present in the :class:`~mne.io.Raw` object.

source_id = uuid.uuid4().hex
player = PlayerLSL(
    raw,
    chunk_size=200,
    name="tutorial-annots",
    source_id=source_id,
    annotations=True,
).start()

# %%
# We can now acquire both streams with 2 :class:`~mne_lsl.stream.StreamLSL` objects.

stream = StreamLSL(2, name="tutorial-annots", source_id=source_id)
stream.connect(acquisition_delay=0.1, processing_flags="all")
stream.info

# %%

stream_annotations = StreamLSL(2, stype="annotations", source_id=source_id)
stream_annotations.connect(acquisition_delay=0.1, processing_flags="all")
stream_annotations.info

# %%
# We can now acquire new samples from both streams and create a matplotlib figure to
# plot the signal and the annotations in real-time.

if not plt.isinteractive():
    plt.ion()
fig, ax = plt.subplots()
plt.show()

n = 0  # number of annotations
colors = ["lightcoral", "lightgreen"]
while n <= 10:
    if stream.n_new_samples == 0:
        continue

    data, ts = stream.get_data(winsize=stream.n_new_samples)
    ax.plot(ts, data[0, :], color="teal")

    if stream_annotations.n_new_samples != 0:
        data_annotations, ts_annotations = stream_annotations.get_data()
        # remove samples with ts=0 which correspond to the empty buffer at the beginning
        n_samples = np.count_nonzero(ts_annotations)
        data_annotations = data_annotations[:, -n_samples:]
        ts_annotations = ts_annotations[-n_samples:]
        # remove annotations outside of the signal range
        sel = np.where((ts[0] <= ts_annotations) & (ts_annotations <= ts[-1]))[0]
        data_annotations = data_annotations[:, sel]
        ts_annotations = ts_annotations[sel]
        # find the position of the annotations in regards to the signal
        idx = np.searchsorted(ts, ts_annotations)
        for sample, time in zip(data_annotations.T, ts[idx]):
            k = np.where(sample != 0)[0][0]  # find the annotation
            ax.axvspan(
                time,
                time + sample[k],
                label=stream_annotations.ch_names[k],
                color=colors[k],
                alpha=0.5,
            )
        n += 1

    fig.canvas.draw()
    fig.canvas.flush_events()

# %%
# Free resources
# --------------
#
# When you are done with a :class:`~mne_lsl.player.PlayerLSL`, a
# :class:`~mne_lsl.stream.StreamLSL` or a :class:`~mne_lsl.stream.EpochsStream` don't
# forget to free the resources they use to continuously mock an LSL stream or
# receive new data from an LSL stream.

stream.disconnect()

# %%

stream_annotations.disconnect()

# %%

player.stop()
