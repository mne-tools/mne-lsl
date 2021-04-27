'''
Format conversion.

It supports converting:

- from .xdf | .gdf | .edf | .bdf | .eeg | .pickle files to mne.io.raw
- from mne.io.raw to .mat.
'''

from .convert2fif import any2fif, pcl2fif, edf2fif, bdf2fif, gdf2fif, xdf2fif, eeg2fif
from .fif2mat import fif2mat
from .load_fif import load_fif_raw, load_fif_multi
from .load_config import load_config