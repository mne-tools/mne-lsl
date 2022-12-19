"""
External packages bundled together with BSL.
"""
import importlib

from ..utils._logs import logger

try:
    psychopy = importlib.import_module(name="psychopy")
    logger.debug("Installed 'psychopy' is functional.")
except (ModuleNotFoundError, ImportError):
    psychopy = importlib.import_module(name=".psychopy", package=__name__)
    logger.debug("Module 'psychopy' has been imported from externals.")
