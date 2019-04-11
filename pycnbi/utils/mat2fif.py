from __future__ import print_function, division, unicode_literals

"""
Convert Matlab signal data into fif format.

Kyuhwa Lee, 2017
Swiss Federal Institute of Technology Lausanne (EPFL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import scipy.io
import numpy as np
import pycnbi.utils.q_common as qc
from pycnbi import logger
import mne

def mat2fif(mat_file, sample_rate, data_field, event_field):
    """
    mat_file: Input Matlab file
    sample_rate: Hz
    data_field: Name of signal
    event_field: Name of event
    """
    data = scipy.io.loadmat(MAT_FILE)
    eeg_data = data[DATA_FIELD]
    event_data = data[EVENT_FIELD]
    num_eeg_channels = eeg_data.shape[0]
    signals = np.concatenate( (event_data, eeg_data), axis=0 )
    assert event_data.shape[1] == eeg_data.shape[1]
    ch_names = ['TRIGGER'] + ['CH%d' % ch for ch in range(num_eeg_channels)]
    ch_info = ['stim'] + ['eeg'] * num_eeg_channels
    info = mne.create_info(ch_names, SAMPLE_RATE, ch_info)
    raw = mne.io.RawArray(signals, info)
    [basedir, fname, fext] = qc.parse_path_list(MAT_FILE)
    fifname = '%s/%s.fif' % (basedir, fname)
    raw.save(fifname, verbose=False, overwrite=True)
    logger.info('Saved to %s.' % fifname)

if __name__ == '__main__':
    MAT_FILE = r'D:\data\Phoneme\data.mat'
    SAMPLE_RATE = 1000.0
    DATA_FIELD = 'ecog' # data containing signals
    EVENT_FIELD = 'phoneme' # data containing events
    mat2fif(MAT_FILE, SAMPLE_RATE, DATA_FIELD, EVENT_FIELD)
