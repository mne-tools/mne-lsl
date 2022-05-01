"""
Sample trigger definition ``.ini`` file.
"""

import os
from pathlib import Path

from ..utils._logs import logger
from ._fetching import _hashfunc, fetch_file

MD5 = "553f24ecb5c8e67ebe597e6e71d7fcdc"
SHA1 = "b841bbba374586434916e70403703388fc67ac74"
URL = "https://github.com/bsl-tools/bsl-datasets/raw/main/trigger_def/trigger_def.ini"  # noqa: E501
PATH = Path("~/bsl_data/trigger_def/trigger_def.ini").expanduser()


def data_path():
    """
    Return the path to an example trigger definition ``.ini`` file.
    If the dataset is not locally present, it is downloaded in the user home
    directory in the folder ``bsl_data/trigger_def``.

    Returns
    -------
    path : Path
        Path to the dataset.
    """
    os.makedirs(PATH.parent, exist_ok=True)

    logger.debug("URL:   %s", URL)
    logger.debug("Hash:  %s", MD5)
    logger.debug("Path:  %s", PATH)

    if PATH.exists() and _hashfunc(PATH, hash_type="md5") == MD5:
        download = False
    elif PATH.exists() and not _hashfunc(PATH, hash_type="md5") == MD5:
        logger.warning(
            "Dataset existing but with different hash. Re-downloading."
        )
        download = True
    else:
        logger.info("Fetching dataset..")
        download = True

    if download:
        fetch_file(URL, PATH, hash_=MD5, hash_type="md5", timeout=10.0)

    return PATH
