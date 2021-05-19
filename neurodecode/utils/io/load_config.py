import os
import sys

from importlib import import_module
from pathlib import Path

#----------------------------------------------------------------------
def load_config(cfg_module):
    """
    Dynamic loading of a config file module.
    
    Parameters
    ----------
    cfg_module : str
        The absolute path to the config file to load 
    """
    if not os.path.isabs(cfg_module):
        cfg_module = os.path.abspath(cfg_module)
        if not os.path.isfile(cfg_module):
            raise Exception('Provide the absolute path to the config file to load.')
    
    cfg_module = Path(cfg_module)
    sys.path.append(str(cfg_module.parent))
    
    return import_module(cfg_module.stem)
