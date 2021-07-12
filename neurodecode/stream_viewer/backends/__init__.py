from .pyqt5 import _BackendPyQt5
try:
    from .vispy import _BackendVispy
except ImportError:
    pass
