"""Qt 6 stream viewer.

This subpackage is never imported by :mod:`mne_lsl` itself: it is imported lazily by
``mne-lsl viewer`` (:mod:`mne_lsl._commands.viewer`) and by
:meth:`mne_lsl.stream.BaseStream.plot`, so ``import mne_lsl`` never pulls Qt in.
Importing it runs the runtime guards below.
"""

import os

from ._bootstrap import (
    _ensure_not_free_threaded,
    _ensure_qt_binding,
    _ensure_qt_stack,
)

# (1) Qt and pyqtgraph are not free-threading safe, while the rest of mne-lsl is.
_ensure_not_free_threaded()
# (2) fail with an actionable message when no Qt 6 binding is installed.
_api_name = _ensure_qt_binding()
# (3) fail with the same message when the binding is installed but the rest of the extra
# is not, which 'pip install mne-lsl' followed by 'pip install PyQt6' produces.
_ensure_qt_stack(_api_name)
# (4) pyqtgraph resolves its binding through its own shim, not through qtpy. Pinning
# PYQTGRAPH_QT_LIB from qtpy's resolved binding makes the two agree by construction
# (an explicit user value wins, and 'assert_binding_coherence' catches a mismatch).
os.environ.setdefault("PYQTGRAPH_QT_LIB", _api_name)

from ._viewer import Viewer  # noqa: E402

__all__ = ("Viewer",)
