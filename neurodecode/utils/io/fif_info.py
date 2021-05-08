from __future__ import print_function, division

import sys
import mne
from builtins import input

import neurodecode.utils.io as io

mne.set_log_level('ERROR')

#----------------------------------------------------------------------
def fif_info(fif_file):
    """
    Output the raw info of .fif file.
    
    PArameters
    ----------
    fif_file : str
        The path to the .fif file
    """
    print('Loading "%s"' % fif_file)
    raw, events = io.load_fif_raw(fif_file)
    print('Raw info: %s' % raw)
    print('Channels: %s' % ', '.join(raw.ch_names))
    print('Events: %s' % set(events[:, 2]))
    print('Sampling freq: %.3f Hz' % raw.info['sfreq'])

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    fif_file = None
    
    if len(sys.argv) > 2:
        raise IOError("Too many arguments provided. Max is 1: fif_file")
    
    if len(sys.argv) == 2:
        fif_file = sys.argv[1]
    
    if len(sys.argv) == 1:
        fif_file = input('Provide the .fif file path?\n>> ')
    
    fif_info(fif_file)
