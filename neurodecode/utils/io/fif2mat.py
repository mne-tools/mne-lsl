from __future__ import print_function, division

"""
Export fif data to mat files.
"""

import neurodecode.utils.q_common as qc
import neurodecode.utils.pycnbi_utils as pu

from neurodecode.utils.io import load_fif_raw
from neurodecode import logger
from scipy.io import savemat

#----------------------------------------------------------------------
def fif2mat(data_dir, out_dir=None):
    """
    Convert MNE format (.fif) to MatLab format (.mat).
    
    Parameters
    ----------
    data_dir : str
        The directory containing the .fif files to convert
    out_dir : str
        The output directory
    """
    # mat file will be in a subfolder
    if not out_dir:
        out_dir = '%s/mat_files' % data_dir
    
    #  create the output directory
    qc.make_dirs(out_dir)
    
    # Convert each fif file 
    for rawfile in qc.get_file_list(data_dir, fullpath=True):
        # keep only fif files
        if rawfile[-4:] != '.fif': continue
        
        # Load fif
        raw, events = pu.load_fif_raw(rawfile)
        
        # Formating
        events[:,0] += 1            # MATLAB uses 1-based indexing
        sfreq = raw.info['sfreq']
        data = dict(signals=raw._data, events=events, sfreq=sfreq, ch_names=raw.ch_names)
        
        # Save
        fname = qc.parse_path(rawfile).name
        matfile = '%s/%s.mat' % (out_dir, fname)
        savemat(matfile, data)
        logger.info('Exported to %s' % matfile)
    
    logger.info('Finished exporting.')

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    out_dir = None
    
    if len(sys.argv) > 3:
        raise IOError("Two many arguments provided, max is 2 (fif_dir, output_dir)")
    
    if len(sys.argv) > 2:
        out_dir  = sys.argv[1]    

    if len(sys.argv) > 1:
        fif_dir  = sys.argv[1]
    
    if len(sys.argv) == 1:
        fif_dir = input('Provide the directory with the fif file to convert: \n>> ')
    
    fif2mat(fif_dir)
