from __future__ import print_function, division

"""
Add Cybex events for patient 6.

* For patients 2 and 5, use mat2fif.py since the mat file structure is completely different.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)
"""

import sys
import mne

from neurodecode.utils.io import load_fif_raw
from builtins import input

mne.set_log_level('ERROR')

def run(fif_file):
    print('Loading "%s"' % fif_file)
    raw, events = load_fif_raw(fif_file)
    print('Raw info: %s' % raw)
    print('Channels: %s' % ', '.join(raw.ch_names))
    print('Events: %s' % set(events[:, 2]))
    print('Sampling freq: %.3f Hz' % raw.info['sfreq'])

# for batch scripts
def batch_run(fif_file=None):
    if not fif_file:
        fif_file = input('fif file name?')
    run(fif_file)

# invoked directly from console
if __name__ == '__main__':
    fif_file = None
    if len(sys.argv) >= 2:
        fif_file = sys.argv[1]
    batch_run(fif_file)
