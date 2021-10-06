"""
Sample dataset of a 64 channel resting-state EEG recording of 2 seconds using
an ANT Neuro amplifier.
"""

import os
from pathlib import Path

from ._fetching import fetch_file, _hashfunc
from .. import logger


MD5 = 'e808b98c464f6d28d0343a054f35c13e'
SHA1 = 'd14dc86d799e6b140d88643282f4229187caa34e'
URL = 'https://github.com/bsl-tools/bsl-datasets/raw/main/eeg_sample/resting_state_short-raw.fif'
PATH = Path('~/bsl_data/eeg_sample/resting_state_short-raw.fif').expanduser()


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
