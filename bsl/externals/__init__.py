"""
External packages bundle together with BSL.
"""
import importlib
from ctypes import CDLL


try:
    # is pylsl installed?
    pylsl = importlib.import_module(name='pylsl')
    # does it find a liblsl?
    libpaths = pylsl.pylsl.find_liblsl_libraries()
    while True:
        try:
            CDLL(next(libpaths))
        except StopIteration:
            raise StopIteration
        except Exception:
            continue
        break
    # clean-up
    del libpaths
except (ModuleNotFoundError, ImportError, StopIteration):
    pylsl = importlib.import_module(name='.pylsl', package=__name__)
