"""
Sample trigger definition file.
"""

import os
from pathlib import Path

from ._fetching import fetch_file, _hashfunc
from .. import logger


MD5 = '96779ac8bd12f70ea01894f86429e27f'
SHA1 = 'b35b7c7e8bc59449b066bc61735be9ee0813218b'
URL = 'https://github.com/bsl-tools/bsl-datasets/raw/main/trigger_def/trigger_def.ini'
PATH = Path('~/bsl_data/trigger_def/trigger_def.ini').expanduser()


def data_path():
    """
    Return the path to the sample trigger event file.
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
