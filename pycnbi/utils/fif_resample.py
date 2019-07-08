from __future__ import print_function, division

"""
Change the sampling frequency of fif files.
Appropriate low-pass filter is applied before resampling.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

import sys
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from builtins import input
from pycnbi import logger

def fif_resample(fif_dir, sfreq_target):
    out_dir = fif_dir + '/fif_resample%d' % sfreq_target
    qc.make_dirs(out_dir)
    for f in qc.get_file_list(fif_dir):
        pp = qc.parse_path(f)
        if pp.ext != 'fif':
            continue
        logger.info('Resampling %s' % f)
        raw, events = pu.load_raw(f)
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
