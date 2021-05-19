from __future__ import print_function
from builtins import input

import os
import mne

from neurodecode import logger
import neurodecode.utils.io as io

#----------------------------------------------------------------------
def fix_channel_names(fif_dir, new_channel_names):
    '''
    Change the channel names of all fif files in a given directory.

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files
    new_channel_names : list
        The list of the new channel names
    '''
    
    for f in io.get_file_list(fif_dir):
        if not io.parse_path(f).ext == 'fif':
            continue
        
        if not os.path.isdir('%s/corrected' % fif_dir):
            io.make_dirs('%s/corrected' % fif_dir)
        
        logger.info('\nLoading %s' % f)
        pp = io.parse_path(f)
        
        raw, eve = io.load_fif_raw(f)
        
        if len(raw.ch_names) != len(new_channel_names):
            raise RuntimeError('The number of new channels do not match that of fif file.')
        
        mapping = {raw.info['ch_names'][k]: new_ch for k, new_ch in enumerate(new_channel_names)}
        mne.rename_channels(raw.info, mapping)
            
        out_fif = '%s/corrected/%s.fif' % (pp.dir, pp.name)
        logger.info('Exporting to %s' % out_fif)
        raw.save(out_fif)
    
    else:
        logger.warning('No fif files found in %s' % fif_dir)

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) > 3:
        raise IOError("Too many arguments provided. Max is 2: fif_dir; new_channel_names")
    
    if len(sys.argv) == 3:
        fif_dir = sys.argv[1]
        new_channel_names =  sys.argv[2]
    
    if len(sys.argv) == 2:
        fif_dir = sys.argv[1]
        new_channel_names = list(input('New channel names (list)? \n>> '))
        
    if len(sys.argv) == 1:
        fif_dir = input('Directory path containing the file files? \n>> ')
        new_channel_names = list(input('New channel names (list)? \n>> '))
        
    fix_channel_names(fif_dir, new_channel_names)

    print('Finished.')
