from __future__ import print_function, division

"""
Export fif data into mat files.

"""

import mne
import scipy.io
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from pycnbi import logger

def fif2mat(data_dir):
    out_dir = '%s/mat_files' % data_dir
    qc.make_dirs(out_dir)
    for rawfile in qc.get_file_list(data_dir, fullpath=True):
        if rawfile[-4:] != '.fif': continue
        raw, events = pu.load_raw(rawfile)
        events[:,0] += 1 # MATLAB uses 1-based indexing
        sfreq = raw.info['sfreq']
        data = dict(signals=raw._data, events=events, sfreq=sfreq, ch_names=raw.ch_names)
        fname = qc.parse_path(rawfile).name
        matfile = '%s/%s.mat' % (out_dir, fname)
        scipy.io.savemat(matfile, data)
        logger.info('Exported to %s' % matfile)
    logger.info('Finished exporting.')

if __name__ == '__main__':
    # path to fif file(s)
    data_dir = r'D:\data\Records\fif'
    fif2mat(data_dir)
