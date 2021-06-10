# -*- coding: utf-8 -*-
from __future__ import print_function, division
from builtins import input

"""
Measure multitaper computation speed.

i7-8700K: 9.9 ms (101.0 Hz)

@author: leeq
"""

import os
import mne
import numpy as np

from neurodecode import logger
from neurodecode.utils.timer import Timer

os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

#----------------------------------------------------------------------
def benchmark_multitaper(nb_ch, sfreq, freq_range, s_len, nb_iter):
    """
    Assess PSD computation speed with mne multitaper.
    
    Parameters
    ----------
    nb_ch : int
        The channels' number
    sfreq : float
        The sampling frequency
    freq_range : list
        The PSD frequencies range [f_min, f_max]
    s_len : float
        The signal's length [secs]
    nb_iter : int
        The number of iterations.
    """
    # Create fake signal
    signal = np.random.rand(nb_ch, int(np.round(sfreq * s_len)))
    
    # Instance a PSD estimator from mne
    psde = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=freq_range[0],\
        fmax=freq_range[1], bandwidth=None, adaptive=False, low_bias=True,\
        n_jobs=1, normalization='length', verbose=None)

    # timer for computation speed assessment
    tm = Timer()
    times = []
    
    # Assess over several iterations
    for i in range(nb_iter):
        tm.reset()
        psde.transform(signal.reshape((1, signal.shape[0], signal.shape[1])))
        times.append(tm.msec())
        
        if nb_iter > 100 and i % 100 == 0:
            logger.info('%d / %d' % (i, nb_iter))
    
    ms = np.mean(times)
    fps = 1000 / ms
    logger.info('Average = %.1f ms (%.1f Hz)' % (ms, fps))
    
 
 #---------------------------------------------------------------------- 
if __name__ == '__main__':
    import sys
    freq_range = [0, 0]
    
    if len(sys.argv) > 6:
        raise IOError("Two many arguments provided, max is 2  (decoder_file and num_decode)")
    
    elif len(sys.argv) == 6:
        nb_ch = sys.argv[0]
        sfreq = sys.argv[1]
        freq_range[0] = sys.argv[2]
        freq_range[1] = sys.argv[3]
        s_len = sys.argv[4]
        nb_iter = sys.argv[5]
        
    elif len(sys.argv) == 1:    
        nb_ch = int(input('How many channels?\n>> '))
        sfreq = float(str(input('Sampling frequency?\n>> ')))
        freq_range[0] = float(str(input('Min PSD frequency?\n>> ')))
        freq_range[1] = float(str(input('Max PSD frequency?\n>> ')))
        s_len = float(str(input('Windows length in secs?\n>> ')))
        nb_iter = int(str(input('Max estimation iter?\n>> ')))
    
    benchmark_multitaper(nb_ch, sfreq, freq_range, s_len, nb_iter)