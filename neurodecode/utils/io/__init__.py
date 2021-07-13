"""
Module containing functions for processing files/directories, loading/saving
data and converting to other formats.

For converting to '.fif', it supports MNE supported formats to '.fif'.
c.f. https://mne.tools/stable/generated/mne.io.read_raw.html

For exporting, it supports:
    - EEGLAB: '.set'
"""

from .convert2fif import pcl2fif, any2fif, dir_any2fif
from .export import write_set, dir_write_set
from .read_raw_fif import read_raw_fif, read_raw_fif_multi
