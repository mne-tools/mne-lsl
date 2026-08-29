"""Runtime guards, application creation and the Qt-ADS binding shim.

Qt is imported lazily inside the functions: :mod:`mne_lsl.viewer` imports this module
before it knows whether a Qt binding is installed, so importing it must never import
:mod:`qtpy`.
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
import threading
from typing import TYPE_CHECKING

from ..utils.logs import logger

if TYPE_CHECKING:
    from types import ModuleType, TracebackType

    from qtpy.QtWidgets import QApplication

_INSTALL_HINT = (
    "Install one of the complete Qt stacks: 'pip install mne-lsl[pyqt6]' or "
    "'pip install mne-lsl[pyside6]'."
)
# Runtime dependencies of the viewer besides the Qt binding itself, i.e. the rest of the
# 'pyqt6' / 'pyside6' extra. The Qt-ADS distribution is binding-specific and is added by
# '_ensure_qt_stack'.
_STACK_MODULES = ("darkdetect", "pyqtgraph", "qtawesome", "superqt")
# Qt-ADS module name per Qt binding, see 'import_ads'.
_ADS_MODULES = {"PyQt6": "PyQt6Ads", "PySide6": "PySide6QtAds"}

# Strong reference to the application created by 'ensure_application'. PyQt6/sip
# destroys the underlying C++ application together with its last Python reference, so a
# caller which discards the returned object would crash the next Qt object built
# ('Must construct a QApplication before a QWidget'). pyqtgraph.mkQApp keeps an
# equivalent module-global for this reason.
_app: QApplication | None = None


def _ensure_not_free_threaded() -> None:
    """Raise if the interpreter is a free-threaded (GIL-disabled) CPython build."""
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    raise RuntimeError(
        "'mne_lsl.viewer' requires a Python build with the GIL: Qt, pyqtgraph and "
        "their bindings are not free-threading safe. This interpreter is a "
        f"free-threaded build of Python {sys.version.split()[0]} (the GIL is currently "
        f"{'enabled' if gil else 'disabled'}). Run the viewer on a regular CPython "
        "build; the rest of mne-lsl supports free-threaded Python."
    )


def _ensure_qt_binding() -> str:
    """Import qtpy and return the name of the resolved Qt binding."""
    try:
        import qtpy
    except ImportError as error:  # includes qtpy.QtBindingsNotFoundError
        raise ImportError(
            "'mne_lsl.viewer' requires a Qt 6 binding, which is missing. "
            f"{_INSTALL_HINT}"
        ) from error
    return qtpy.API_NAME


def _ensure_qt_stack(api_name: str) -> None:
    """Raise if the Qt binding is installed but the rest of the extra is not.

    Parameters
    ----------
    api_name : str
        Name of the Qt binding qtpy resolved, e.g. ``'PyQt6'``.

    Raises
    ------
    ImportError
        If a module of the viewer's Qt stack is missing, naming every missing one.

    Notes
    -----
    ``pip install mne-lsl`` followed by ``pip install PyQt6`` is a common half-install,
    and the binding guard passes on it: importing the viewer then failed on whichever
    module happened to be reached first, e.g. ``No module named 'PyQt6Ads'``, naming
    neither ``mne-lsl`` nor the extra which provides it.

    The modules are looked up rather than imported: this runs on the import path of
    every entry point, and importing the stack twice is neither free nor this
    function's job.
    """
    ads = _ADS_MODULES.get(api_name)
    modules = _STACK_MODULES if ads is None else (*_STACK_MODULES, ads)
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if not missing:
        return
    raise ImportError(
        f"'mne_lsl.viewer' requires {', '.join(sorted(missing))}, which the "
        f"environment does not provide, although the Qt binding {api_name} is "
        f"installed. {_INSTALL_HINT}"
    )


def assert_binding_coherence() -> None:
    """Raise if qtpy and pyqtgraph resolved different Qt bindings."""
    import pyqtgraph as pg
    import qtpy

    if pg.Qt.QT_LIB != qtpy.API_NAME:
        raise RuntimeError(
            f"Qt binding mismatch: qtpy resolved {qtpy.API_NAME} while pyqtgraph "
            f"resolved {pg.Qt.QT_LIB}. Importing 'mne_lsl.viewer' sets "
            "'PYQTGRAPH_QT_LIB' from qtpy, thus this mismatch comes from an explicit "
            "'QT_API'/'PYQTGRAPH_QT_LIB' pair or from pyqtgraph being imported first."
        )


def ensure_application(name: str = "mne-lsl viewer") -> QApplication:
    """Return the running ``QApplication``, creating one if needed.

    Parameters
    ----------
    name : str
        Application name, set on the returned application.

    Returns
    -------
    app : QApplication
        The reused or newly created application instance.
    """
    from qtpy.QtWidgets import QApplication

    global _app
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(name)
    # applied whether the application was created or reused, as the policy is a property
    # of the process and not of the application object.
    install_exception_policy()
    _app = app  # keep a strong reference alive, see the module-level comment
    return app


def _excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> None:
    """Log an unhandled exception of the main thread, then return.

    Parameters
    ----------
    exc_type : type
        Class of the unhandled exception.
    exc_value : BaseException
        The unhandled exception.
    exc_tb : traceback | None
        Traceback of the unhandled exception.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Ctrl+C while the event loop runs surfaces at the next bytecode, which is often
        # inside a slot. Printing alone would strand the viewer, as
        # 'sys.__excepthook__' never exits, so the application is asked to quit.
        # Delegation targets the *constant* '__excepthook__', never a previously
        # installed hook, which is what keeps the policy idempotent.
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.quit()
        return
    # CRITICAL, not ERROR: an unhandled exception must not become invisible under
    # 'MNE_LSL_LOG_LEVEL=CRITICAL', which is exactly the silent no-op this policy exists
    # to prevent. 'SystemExit' is deliberately not special-cased -- CPython handles it
    # before 'sys.excepthook', and PyQt6 intercepts it from a slot, so that branch is
    # unreachable.
    logger.critical(
        "Unhandled exception in the viewer", exc_info=(exc_type, exc_value, exc_tb)
    )


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Log an unhandled exception of a worker thread, then return.

    Parameters
    ----------
    args : threading.ExceptHookArgs
        The named tuple provided by :mod:`threading`, holding the exception triplet and
        the thread which raised it.
    """
    if args.exc_type is SystemExit:
        return  # 'threading.__excepthook__' ignores it too, and only the exact class
    logger.critical(
        "Unhandled exception in the viewer thread %r",
        args.thread.name if args.thread is not None else None,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def install_exception_policy() -> None:
    """Install a consistent unhandled-exception policy across the Qt bindings.

    An unhandled exception raised inside a slot behaves differently per binding: PyQt6
    calls ``qFatal()`` and aborts the process with SIGABRT, while PySide6 prints the
    traceback and keeps running. The same bug is therefore a hard crash on one binding
    and a survivable, unlogged event on the other.

    Notes
    -----
    Replacing :data:`sys.excepthook` is sufficient to obtain one behaviour on both
    bindings, the traceback logged at the CRITICAL level and the event loop kept alive.
    PyQt6 aborts only while the hook is *identically* :data:`sys.__excepthook__`, thus
    assigning any other callable, even one which merely re-implements the default,
    disarms the abort. No slot-wrapping decorator is needed. Verified across a matrix of
    raise sites -- timer slot, ``eventFilter``, ``paintEvent``, a nested ``exec()``,
    :class:`~qtpy.QtCore.QThread` -- and exception classes, on both bindings.

    A :class:`KeyboardInterrupt` surfacing inside a slot becomes a non-event: the
    traceback is printed and the event loop continues. The delegation below keeps Ctrl+C
    normal on the console path only.

    The hooks are module-level functions and the previous hook is never captured, thus
    re-installing the policy is a no-op. This matters because :func:`ensure_application`
    calls it on every invocation.

    One divergence a hook cannot equalize: an exception raised in a *synchronous*
    Python -> C++ -> Python round trip, e.g. ``app.sendEvent()`` reaching a virtual
    override, is re-raised at the calling frame under PySide6 while it is hook-handled
    under PyQt6. Only the caller's own ``try/except`` covers that case.
    """
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


def configure_docking() -> None:
    """Set the process-wide Qt Advanced Docking System configuration flags.

    Must be called before the first ``CDockManager`` is constructed: the flags are
    static and the constructor consumes them. Idempotent, and safe to call again
    afterwards as long as the values do not change.

    Notes
    -----
    ``FocusHighlighting`` is not merely read at construction: a manager built while it
    was off, with the flag switched on afterwards, crashes the process on the next
    ``addDockWidget`` -- the constructor builds the focus controller which the code
    paths guarded by the flag then dereference. The two XML flags keep ``saveState()`` a
    readable byte string rather than the zlib-compressed default, which is what a saved
    configuration stores.
    """
    ads = import_ads()
    flags = ads.CDockManager.eConfigFlag
    for flag, value in (
        (flags.FocusHighlighting, True),
        (flags.EqualSplitOnInsertion, True),
        (flags.XmlCompressionEnabled, False),
        (flags.XmlAutoFormattingEnabled, True),
    ):
        ads.CDockManager.setConfigFlag(flag, value)


def import_ads() -> ModuleType:
    """Return the Qt Advanced Docking System module matching the active binding.

    Qt-ADS ships one distribution per binding under a different module name
    (``PyQt6Ads`` for PyQt6, ``PySide6QtAds`` for PySide6). This function is the single
    place in the viewer where that difference is resolved; consumers do
    ``ads = import_ads()`` at module level.

    Always use the **scoped** enum form, e.g. ``ads.DockWidgetArea.TopDockWidgetArea``:
    the flat form used by the PySide6-QtAds examples, e.g. ``ads.TopDockWidgetArea``,
    does not exist in ``PyQt6Ads``, thus only the scoped form works under both bindings.

    Returns
    -------
    ads : module
        The Qt-ADS binding module.
    """
    import qtpy

    if qtpy.PYQT6:
        import PyQt6Ads as ads
    elif qtpy.PYSIDE6:
        import PySide6QtAds as ads
    else:
        raise RuntimeError(
            f"'mne_lsl.viewer' requires PyQt6 or PySide6, while qtpy resolved "
            f"{qtpy.API_NAME}. {_INSTALL_HINT}"
        )
    return ads
