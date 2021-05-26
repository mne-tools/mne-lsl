'''
Module containing functions for processing files/directories, loading/saving data and converting to other formats.

For converting, it supports:

- from .xdf | .gdf | .edf | .bdf | .eeg | .pickle | .mat files to mne.io.raw
- from mne.io.raw to .mat.

'''

from .convert2fif import any2fif, pcl2fif, edf2fif, bdf2fif, gdf2fif, xdf2fif, eeg2fif, event_timestamps_to_indices
from .export import write_set, dir_write_set
from .io_file_dir import get_file_list, get_dir_list, make_dirs
from .load_config import load_config
from .load_mat import load_mat
from .read_raw_fif import read_raw_fif, read_raw_fif_multi
