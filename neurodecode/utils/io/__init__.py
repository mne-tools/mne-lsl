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
from .io_file_dir import get_file_list, get_dir_list, make_dirs, save_obj, load_obj, loadtxt_fast, parse_path, forward_slashify
from .load_mat import load_mat