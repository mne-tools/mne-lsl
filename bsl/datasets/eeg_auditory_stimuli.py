"""
Sample dataset of a 64 channel EEG recording of 4 minutes including 75 rest
events of 1 second and 75 auditory stimuli events of 0.8 seconds.
"""

import os
from pathlib import Path

from ._fetching import fetch_file, _hashfunc
from .. import logger


MD5 = ''
SHA1 = ''
URL = 'https://github.com/bsl-tools/bsl-datasets/raw/main/eeg_sample/auditory_stimuli-raw.fif'
PATH = Path('~/bsl_data/eeg_sample/auditory_stimuli-raw.fif').expanduser()


def data_path():
    """
    Return the path to the sample dataset.
    If the dataset is not locally present, it is downloaded in the user home
    directory in the folder ``bsl_data``.
    """
    os.makedirs(PATH.parent, exist_ok=True)

    logger.debug('URL:   %s' % (URL,))
    logger.debug('Hash:  %s' % (MD5,))
    logger.debug('Path:  %s' % (PATH,))

    if PATH.exists() and _hashfunc(PATH, hash_type='md5') == MD5:
        download = False
    elif PATH.exists() and not _hashfunc(PATH, hash_type='md5') == MD5:
        logger.warning(
            'Dataset existing but with different hash. Re-downloading.')
        download = True
    else:
        logger.info('Fetching dataset..')
        download = True

    if download:
        fetch_file(URL, PATH, hash_=MD5, hash_type='md5', timeout=10.)

    return PATH
