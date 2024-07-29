"""
Real-time evoked responses
==========================

With a :class:`~mne_lsl.stream.EpochsStream`, we can build a real-time evoked response
vizualization. This is useful to monitor the brain activity in real-time.
"""

from mne.datasets import sample
from mne.io import read_raw_fif

# dataset used in the example
data_path = sample.data_path()
fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
raw = read_raw_fif(fname, preload=False).pick("grad").load_data()
