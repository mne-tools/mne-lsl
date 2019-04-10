from __future__ import print_function, division

"""
Compress feedback images into a single pickle object.

"""

import gzip
import pycnbi.utils.q_common as qc
from pycnbi.protocols.viz_human import read_images
from pycnbi import logger
try:
    import cPickle as pickle  # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle  # Python 3 (C version is the default)

if __name__ == '__main__':
    LEFT_IMAGE_DIR = r'D:\work\pycnbi_protocols\BodyFeedback\left_behind'
    RIGHT_IMAGE_DIR = r'D:\work\pycnbi_protocols\BodyFeedback\right_behind'
    EXPORT_IMAGE_DIR = r'D:\work\pycnbi_protocols\BodyFeedback'

    if pickle.HIGHEST_PROTOCOL >= 4:
        outfile = '%s/BodyVisuals.pkl' % EXPORT_IMAGE_DIR
        tm = qc.Timer()
        logger.info('Reading images from %s' % LEFT_IMAGE_DIR )
        left_images = read_images(LEFT_IMAGE_DIR)
        logger.info('Reading images from %s' % RIGHT_IMAGE_DIR)
        right_images = read_images(RIGHT_IMAGE_DIR)
        logger.info('Took %.1f s. Start exporting ...' % tm.sec())
        img_data = {'left_images':left_images, 'right_images':right_images}
        with gzip.open(outfile, 'wb') as fp:
            pickle.dump(img_data, fp)
        logger.info('Exported to %s' % outfile)
    else:
        logger.warning('Your Python pickle protocol version is less than 4, which will be slower with loading a pickle object.')
