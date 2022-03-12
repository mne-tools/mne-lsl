"""
External packages bundled together with BSL.
"""
import importlib

from ..utils._logs import logger

try:
    pylsl = importlib.import_module(name='pylsl')
    logger.debug("Installed 'pylsl' is functional.")
except (ModuleNotFoundError, ImportError, RuntimeError):
    pylsl = importlib.import_module(name='.pylsl', package=__name__)
    logger.debug("Module 'pylsl' has been imported from externals.")
