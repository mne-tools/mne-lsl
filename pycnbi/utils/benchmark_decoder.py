# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Decoder speed test.

Results on i7-8700K machine:
1000 Hz, 6 channels, 500-sample window:
  sklearn 0.17.1(pip) = 32.7 Hz, 0.19.1(conda) = 28.8 Hz
 with intel-mkl library:
  sklearn 0.17(pip) = 25.0 Hz, 0.19.1(conda) = 24.0 Hz

512 Hz, 64 channels, 256-sample window:
  sklearn 0.17(pip) = 27.9 Hz, 0.19.1(conda) = 24.0 Hz

@author: leeq
"""

import pycnbi
import numpy as np
import pycnbi.utils.q_common as qc
from pycnbi.decoder.decoder import BCIDecoder, BCIDecoderDaemon

if __name__ == '__main__':
    decoder_file = 'PATH_TO_CLASSIFIER_FILE'
    decoder = BCIDecoder(decoder_file, buffer_size=1.0)
    num_decode = 200
    tm = qc.Timer()
    times = []
    while len(times) < num_decode:
        tm.reset()
        prob = decoder.get_prob()
        times.append(tm.msec())
        if len(times) % 10 == 0:
            print(len(times), end=' ')

    ms = np.mean(times)
    fps = 1000 / ms
    print('\nAverage = %.1f ms (%.1f Hz)' % (ms, fps))
