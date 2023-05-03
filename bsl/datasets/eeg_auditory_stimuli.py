import os
from pathlib import Path

from ..utils.logs import logger
from ._fetching import _hashfunc, fetch_file

MD5 = "ea0d40643bdc1c88e2b808c7128a0eba"
SHA1 = "d86e0ad1d53224a41e06b918e5d709615e2da32a"
URL = "https://github.com/bsl-tools/bsl-datasets/raw/main/eeg_sample/auditory_stimuli-raw.fif"  # noqa: E501
PATH = Path("~/bsl_data/eeg_sample/auditory_stimuli-raw.fif").expanduser()


def data_path():  # noqa
    """
    Path to a sample EEG dataset recorded on ANT Neuro Amplifier with 64
    electrodes. The recording last 184 seconds and include 75 rest events (1)
    lasting 1 second and 75 auditory stimuli events (4) lasting 0.8 second.
    If the dataset is not locally present, it is downloaded in the user home
    directory in the folder ``bsl_data/eeg_sample``.

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
        logger.warning("Dataset existing but with different hash. Re-downloading.")
        download = True
    else:
        logger.info("Fetching dataset..")
        download = True

    if download:
        fetch_file(URL, PATH, hash_=MD5, hash_type="md5", timeout=10.0)

    return PATH
