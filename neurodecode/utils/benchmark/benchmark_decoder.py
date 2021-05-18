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
import numpy as np
from neurodecode.utils.timer import Timer
from neurodecode.decoder.decoder import BCIDecoder

#----------------------------------------------------------------------
def benchmark_BCIdecoder(decoder_file, num_decode=200):
    """
    Assess the BCI decoder speed performance.
    
    Parameters
    ----------
    decoder_file : str
        The path to the decoder file.
    num_decode : int
        The number of decoding trials for future averaging.
    """
    decoder = BCIDecoder(decoder_file, buffer_size=1.0)
    
    tm = Timer()
    times = []
    
    while len(times) < num_decode:
        tm.reset()
    
        decoder.get_prob()
        times.append(tm.msec())
    
        if len(times) % 10 == 0:
            print(len(times), end=' ')

    # compute avegare classification per sec
    ms = np.mean(times)
    cps = 1000 / ms
    
    return ms, cps

if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) > 3:
        raise IOError("Two many arguments provided, max is 2  (decoder_file and num_decode)")
    
    if len(sys.argv) == 3:
        num_decode = sys.argv[2]
        decoder_file = sys.argv[1]
    
    if len(sys.argv) == 2:
        num_decode = input('How many decoding trials?\n>> ')
        decoder_file = sys.argv[1]
    
    if len(sys.argv) == 1:    
        num_decode = input('How many decoding trials?\n>> ')
        decoder_file = str(input('Provide the path to the decoder file:\n>> '))
            
    ms, cps = benchmark_BCIdecoder(decoder_file, num_decode)
    print('\nAverage = %.1f ms (%.1f Hz)' % (ms, cps))    
