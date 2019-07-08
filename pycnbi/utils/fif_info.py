from __future__ import print_function, division

"""
Add Cybex events for patient 6.

* For patients 2 and 5, use mat2fif.py since the mat file structure is completely different.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)
"""

import sys
import mne
import numpy as np
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from builtins import input
from IPython import embed
mne.set_log_level('ERROR')

def run(fif_file):
    print('Loading "%s"' % fif_file)
    raw, events = pu.load_raw(fif_file)
    print('Raw info: %s' % raw)
    print('Channels: %s' % ', '.join(raw.ch_names))
    print('Events: %s' % set(events[:, 2]))
    print('Sampling freq: %.3f Hz' % raw.info['sfreq'])
    qc.print_c('\n>> Interactive mode start. Type quit or Ctrl+D to finish', 'g')
    qc.print_c('>> Variables: raw, events\n', 'g')
    embed()

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
