from __future__ import print_function, division
from builtins import input

import sys

from neurodecode import logger
import neurodecode.utils.io as io

#----------------------------------------------------------------------
def fif_resample(fif_dir, sfreq_target, out_dir=None):
    """
    Resample all .fif file contained in a directory and save them.
        
    Parameters
    ----------
    fif_dir : str
        The absolute path to the directory containing the .fif files to resample
    sfreq_target : float
        Tne desired sampling rate
    out_dir : str
        The absolute path to the directory where the new files will be saved.
        If None, they will be saved in %fif_dir%/fif_resample%sfreq_target%
    """
    if out_dir is None:
        out_dir = fif_dir + '/fif_resample_%d' % sfreq_target
    
    io.make_dirs(out_dir)
    
    for f in io.get_file_list(fif_dir):
        pp = io.parse_path(f)
        
        if pp.ext != 'fif':
            continue
        
        logger.info('Resampling %s' % f)
        
        raw, events = io.load_fif_raw(f)
        raw.resample(sfreq_target)
        
        fif_out = '%s/%s.fif' % (out_dir, pp.name)
        raw.save(fif_out)
        logger.info('Exported to %s' % fif_out)

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    if len(sys.argv) > 3:
        raise IOError("Too many arguments provided. Max is 2: fif_dir; sfreq_target")
    
    if len(sys.argv) == 3:
        fif_dir = sys.argv[1]
        sfreq_target =  sys.argv[2]
    
    if len(sys.argv) == 2:
        fif_dir = sys.argv[1]
        sfreq_target = float(input('Target sampling frequency? \n>> '))
        
    if len(sys.argv) == 1:
        fif_dir = input('Directory path containing the file files? \n>> ')
        sfreq_target = float(input('Target sampling frequency? \n>> '))
        
    fif_resample(fif_dir, sfreq_target)

    print('Finished.')
