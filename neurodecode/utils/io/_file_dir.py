import os
import shutil
from pathlib import Path

from ... import logger


def make_dirs(dirname, delete=False):
    """
    Create a new directory.

    Parameters
    ----------
    dirname : str | Path
        Name of the new directory.
    delete : bool
        If ``True`` and the directory already exists, it will be deleted.
    """
    dirname = Path(dirname)
    if dirname.exists() and delete:
        try:
            shutil.rmtree(dirname)
        except OSError:
            logger.error(
                'Directory was not completely removed. '
                '(Perhaps a Dropbox folder?). Continuing.', exc_info=True)
    if not dirname.exists():
        logger.info(f'Creating: {dirname}')
        os.makedirs(dirname)
