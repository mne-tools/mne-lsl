"""
Sample dataset of a 64 channel resting-state EEG recording of 40 seconds using
an ANT Neuro amplifier.
"""

from pathlib import Path

from ._fetching import fetch_file, _hashfunc
from .. import logger
from ..utils.io._file_dir import make_dirs


MD5 = '8925f81af22390fd17bb3341d553430f'
URL = 'https://github.com/bsl-tools/bsl-datasets/raw/main/eeg/resting-state-sample-raw.fif'


def data_path():
    """
    Return the path to the sample dataset.
    If the dataset is not locally present, it is downloaded in the user home
    directory in the folder ``bsl-datasets``.
    """
    path = Path('~/bsl_data/eeg/resting-state-sample-raw.fif').expanduser()
    make_dirs(path.parent)

    logger.debug('URL:   %s' % (URL,))
    logger.debug('Hash:  %s' % (MD5,))
    logger.debug('Path:  %s' % (path,))

    if path.exists() and _hashfunc(path) == MD5:
        download = False
    elif path.exists() and not _hashfunc(path) == MD5:
        logger.warning(
            'Dataset existing but with different hash. Re-downloading.')
        download = True
    else:
        logger.info('Fetching dataset..')
        download = True

    if download:
        fetch_file(URL, path, hash_=MD5, hash_type='md5', timeout=10.)

    return path
