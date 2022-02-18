"""
Sample EEG dataset recorded on ANT Neuro Amplifier with 64 electrodes. This is
a resting-state recording lasting 40 seconds.
"""

import os
from pathlib import Path

from ._fetching import fetch_file, _hashfunc
from ..utils._logs import logger


MD5 = '8925f81af22390fd17bb3341d553430f'
SHA1 = '65dbf592fa6e18cee049f244ca53c504ddabacc1'
URL = 'https://github.com/bsl-tools/bsl-datasets/raw/main/eeg_sample/resting_state-raw.fif'  # noqa: E501
PATH = Path('~/bsl_data/eeg_sample/resting_state-raw.fif').expanduser()


def data_path():
    """
    Path to a sample EEG dataset recorded on ANT Neuro Amplifier with 64
    electrodes. This is a resting-state recording lasting 40 seconds.
    If the dataset is not locally present, it is downloaded in the user home
    directory in the folder ``bsl_data/eeg_sample``.

    Returns
    -------
    path : Path
        Path to the dataset.
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
