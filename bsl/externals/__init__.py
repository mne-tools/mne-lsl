"""
External packages bundled together with BSL.
"""
import importlib


try:
    pylsl = importlib.import_module(name='pylsl')
except (ModuleNotFoundError, ImportError, RuntimeError):
    pylsl = importlib.import_module(name='.pylsl', package=__name__)
