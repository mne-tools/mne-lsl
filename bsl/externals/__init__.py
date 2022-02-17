"""
External packages bundle together with BSL.
"""
import importlib


try:
    pylsl = importlib.import_module(name='pylsl')
except (ModuleNotFoundError, ImportError):
    pylsl = importlib.import_module(name='.pylsl', package=__name__)
