from __future__ import print_function, division

import gzip
import sys

from neurodecode import logger
from neurodecode.utils.timer import Timer
from neurodecode.protocols.viz_human import read_images

try:
    import cPickle as pickle  # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle  # Python 3 (C version is the default)

#----------------------------------------------------------------------
def images2pkl(in_dir, out_dir):
    """
    Format .png images to .pkl for faster loading
    
    Parameters
    ----------
    in_dir : str
        The path to the directory containing the images
    out_dir : str
        The output directory path 
    """
    if pickle.HIGHEST_PROTOCOL >= 4:
        
        tm = Timer()

        logger.info('Reading images from %s' % in_dir)
        images = read_images(in_dir)

        logger.info('Took %.1f s. Start exporting ...' % tm.sec())

        with gzip.open(out_dir, 'wb') as fp:
            pickle.dump(images, fp)

        logger.info('Exported to %s' % out_dir)

    else:
        logger.warning('Your Python pickle protocol version is less than 4, which will be slower with loading a pickle object.')
    
#----------------------------------------------------------------------
if __name__ == '__main__':
    
    if len(sys.argv) > 3:
        raise IOError("Two many arguments provided, max is 2 (fif_dir, output_dir)")

    if len(sys.argv) == 3:
        in_dir  = sys.argv[1]
        out_dir  = sys.argv[2]    
    
    if len(sys.argv) == 2:
        in_dir  = sys.argv[1]
        out_dir = input(" Output directory \n>> ")
    
    if len(sys.argv) == 1:
        in_dir = input('Provide the directory containing the images to convert: \n>> ')
        out_dir = input(" Output directory \n>> ")
        
    images2pkl(in_dir, out_dir)