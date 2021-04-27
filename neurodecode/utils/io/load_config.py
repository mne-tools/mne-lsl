import sys
import os
from importlib import import_module
from pathlib import Path

#----------------------------------------------------------------------
def load_config(cfg_module):
    """
    Dynamic loading of a config file module.
    
    cfg_module = absolute path to the config file to load 
    """
    cfg_module = Path(cfg_module)
    cfg_path, cfg_name = os.path.split(cfg_module)
    sys.path.append(cfg_path)
    
    return import_module(cfg_name.split('.')[0])
