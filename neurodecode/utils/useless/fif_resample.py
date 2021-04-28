from __future__ import print_function, division

"""
Change the sampling frequency of fif files.
Appropriate low-pass filter is applied before resampling.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

import sys
import neurodecode.utils.q_common as qc

from builtins import input

from neurodecode import logger
from neurodecode.utils.io import load_fif_raw, make_dirs, parse_path, get_file_list


def fif_resample(fif_dir, sfreq_target):
    out_dir = fif_dir + '/fif_resample%d' % sfreq_target
    make_dirs(out_dir)
    for f in get_file_list(fif_dir):
        pp = parse_path(f)
        if pp.ext != 'fif':
            continue
        logger.info('Resampling %s' % f)
        raw, events = load_fif_raw(f)
        raw.resample(sfreq_target)
        fif_out = '%s/%s.fif' % (out_dir, pp.name)
        raw.save(fif_out)
        logger.info('Exported to %s' % fif_out)

# for batch scripts
def batch_run(fif_dir=None, sfreq_target=None):
    if not fif_dir:
        fif_dir = input('Data file path? ')
    if not sfreq_target:
        sfreq_target = input('Target sampling frequency? ')
    sfreq_target = float(sfreq_target)
    fif_resample(fif_dir, sfreq_target)

# invoked directly from console
if __name__ == '__main__':
    if len(sys.argv) < 3:
        fif_dir = input('Data file path? ')
        sfreq_target = float(input('Target sampling frequency? '))
    else:
        fif_dir = sys.argv[1]
        sfreq_target = float(sys.argv[2])
    fif_resample(fif_dir, sfreq_target)
    print('Finished.')
