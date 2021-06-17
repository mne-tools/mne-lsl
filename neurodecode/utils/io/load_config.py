import sys
from pathlib import Path
from importlib import import_module

from ... import logger


def load_config(cfg_module):
    """
    Dynamic loading of a config file module.

    Parameters
    ----------
    cfg_module : str
        The absolute path to the config file to load.
    """
    cfg_module = Path(cfg_module)
    if not cfg_module.is_absolute():
        cfg_module = cfg_module.absolute()
        if not cfg_module.exists():
            logger.error('CFG file not found. Provide the absolute path '
                         'to the config file to load.')
            raise IOError

    sys.path.append(str(cfg_module.parent))

    return import_module(cfg_module.stem)
