from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import subprocess
import sys
import sysconfig
import threading
from typing import TYPE_CHECKING

import pyqtgraph as pg
import pytest
import qtpy
from qtpy.QtWidgets import QApplication, QMainWindow

import mne_lsl.viewer
import mne_lsl.viewer._bootstrap
from mne_lsl.viewer._bootstrap import (
    _ensure_not_free_threaded,
    _ensure_qt_binding,
    _ensure_qt_stack,
    _excepthook,
    _thread_excepthook,
    assert_binding_coherence,
    configure_docking,
    ensure_application,
    import_ads,
    install_exception_policy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Safe at module level: 'tests/viewer/conftest.py' keeps this whole package out of
# collection (via 'collect_ignore_glob') before any module here is ever imported, on a
# free-threaded build or without a Qt binding -- see the comment there.


def _module_names() -> list[str]:
    """Return the name of every module of 'mne_lsl.viewer', subpackages included."""
    return [
        name
        for _, name, _ in pkgutil.walk_packages(
            mne_lsl.viewer.__path__, prefix="mne_lsl.viewer."
        )
    ]


def test_module_names() -> None:
    """Test that the module walk finds the subpackages."""
    names = _module_names()
    assert "mne_lsl.viewer._bootstrap" in names
    for subpackage in ("backend", "controller", "display", "theme", "widgets"):
        assert f"mne_lsl.viewer.{subpackage}" in names, subpackage


@pytest.mark.parametrize("name", _module_names())
def test_import_module(name: str) -> None:
    """Test that every module imports without side effect.

    'tools/stubgen.py' imports every module to generate the stub files, thus a module
    which builds a QApplication, a widget or applies the theme on import would break the
    stub generation.
    """
    importlib.import_module(name)


def test_ensure_not_free_threaded() -> None:
    """Test the free-threaded Python guard."""
    # a free-threaded build never reaches this point, the package is not collected
    assert not sysconfig.get_config_var("Py_GIL_DISABLED")
    _ensure_not_free_threaded()  # no-op on a build with the GIL


def test_ensure_qt_binding() -> None:
    """Test that the Qt binding guard reports the binding resolved by qtpy."""
    assert _ensure_qt_binding() == qtpy.API_NAME
    assert qtpy.API_NAME in ("PyQt6", "PySide6")


def test_ensure_qt_stack() -> None:
    """Test that the running environment provides the whole Qt stack of the viewer.

    The Qt-ADS distribution is binding-specific, thus the guard has to resolve the right
    module name rather than a fixed one.
    """
    _ensure_qt_stack(qtpy.API_NAME)  # a no-op on a complete install
    ads_modules = mne_lsl.viewer._bootstrap._ADS_MODULES
    assert ads_modules[qtpy.API_NAME] == import_ads().__name__


@pytest.mark.parametrize("api_name", ["PyQt6", "PySide6"])
def test_ensure_qt_stack_names_the_extra(
    api_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a half-installed extra raises naming both the module and the extra.

    'pip install mne-lsl' followed by 'pip install PyQt6' passes the binding guard, and
    the import then failed on whichever module it reached first -- 'No module named
    'PyQt6Ads'', which names neither mne-lsl nor the extra which provides it.
    """
    monkeypatch.setattr(
        mne_lsl.viewer._bootstrap, "_STACK_MODULES", ("mne_lsl_viewer_absent",)
    )
    with pytest.raises(ImportError, match="mne_lsl_viewer_absent") as error:
        _ensure_qt_stack(api_name)
    assert "mne-lsl[pyqt6]" in str(error.value)
    assert "mne-lsl[pyside6]" in str(error.value)
    assert api_name in str(error.value)


def test_pyqtgraph_binding() -> None:
    """Test that qtpy and pyqtgraph agree on the Qt binding."""
    assert os.environ["PYQTGRAPH_QT_LIB"] == qtpy.API_NAME
    assert_binding_coherence()


def test_assert_binding_coherence_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a qtpy/pyqtgraph binding mismatch raises."""
    monkeypatch.setattr(pg.Qt, "QT_LIB", "PyQt5")
    with pytest.raises(RuntimeError, match="Qt binding mismatch"):
        assert_binding_coherence()


def test_ensure_application(app: QApplication) -> None:
    """Test that the application is created once and reused."""
    assert ensure_application() is app
    assert QApplication.instance() is app


def test_import_ads() -> None:
    """Test that the Qt-ADS shim resolves the binding-specific module."""
    ads = import_ads()
    assert ads.__name__ in ("PyQt6Ads", "PySide6QtAds")
    assert hasattr(ads, "CDockManager")
    assert hasattr(ads, "CDockWidget")


_FLAGS = (
    ("FocusHighlighting", True),
    ("EqualSplitOnInsertion", True),
    ("XmlCompressionEnabled", False),
    ("XmlAutoFormattingEnabled", True),
)


def test_configure_docking() -> None:
    """Test that the four docking configuration flags are set to their intended value.

    Never written as a flip-and-restore: the flags are process-global static state which
    a 'CDockManager' constructor consumes, and a manager built while 'FocusHighlighting'
    was off segfaults on its next 'addDockWidget' once the flag is switched on.
    """
    ads = import_ads()
    configure_docking()
    for name, value in _FLAGS:
        flag = getattr(ads.CDockManager.eConfigFlag, name)
        assert ads.CDockManager.testConfigFlag(flag) is value, name


def test_configure_docking_is_idempotent(flush_deletes: Callable[..., None]) -> None:
    """Test that a second call, with a manager already built, changes nothing.

    A future edit turning the assignment into a toggle would pass a single-call test and
    take the process down here instead, as the crash is in 'addDockWidget'.
    """
    ads = import_ads()
    configure_docking()
    host = QMainWindow()
    manager = ads.CDockManager(host)
    configure_docking()
    for name, value in _FLAGS:
        flag = getattr(ads.CDockManager.eConfigFlag, name)
        assert ads.CDockManager.testConfigFlag(flag) is value, name
    dock = ads.CDockWidget(manager, "bootstrap-flags")
    manager.addDockWidget(ads.DockWidgetArea.CenterDockWidgetArea, dock)
    assert manager.dockWidgetsMap()["bootstrap-flags"] is dock
    host.close()
    flush_deletes(host)


def test_viewer_public_api() -> None:
    """Test that only 'Viewer' is public and that it is exported."""
    assert mne_lsl.viewer.__all__ == ("Viewer",)
    assert isinstance(mne_lsl.viewer.Viewer, type)


def _exc_info(error: BaseException) -> tuple:
    """Return a real exception triplet for 'error'."""
    try:
        raise error
    except BaseException:  # noqa: B036 -- re-raised through the returned traceback
        return sys.exc_info()


def test_install_exception_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that both hooks are replaced and that re-installing is a no-op.

    Asserted through 'monkeypatch': pytest-qt swaps 'sys.excepthook' at every test
    setup, thus the live global hook is its own, not ours, and must not be read.
    """
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    install_exception_policy()
    # the identity, not the behaviour, is what disarms PyQt6's 'qFatal()' abort.
    assert sys.excepthook is not sys.__excepthook__
    assert threading.excepthook is not threading.__excepthook__
    hooks = (sys.excepthook, threading.excepthook)
    install_exception_policy()
    assert (sys.excepthook, threading.excepthook) == hooks


def test_excepthook(caplog: pytest.LogCaptureFixture) -> None:
    """Test that an unhandled exception is logged and swallowed."""
    caplog.set_level(logging.ERROR, logger="mne_lsl")
    assert _excepthook(*_exc_info(ValueError("boom in a slot"))) is None
    assert "Traceback" in caplog.text
    assert "boom in a slot" in caplog.text
    assert "ValueError" in caplog.text


def test_excepthook_delegates(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that Ctrl+C prints through the interpreter and asks the viewer to quit.

    'SystemExit' is deliberately not covered: CPython handles it before
    'sys.excepthook' runs and PyQt6 intercepts it from a slot, thus a test of that
    branch would assert on behaviour which can never occur.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *args: calls.append(args))
    caplog.set_level(logging.ERROR, logger="mne_lsl")
    _excepthook(*_exc_info(KeyboardInterrupt()))
    assert len(calls) == 1
    assert not caplog.records


def test_thread_excepthook(caplog: pytest.LogCaptureFixture) -> None:
    """Test that an unhandled exception of a worker thread is logged."""
    args = threading.ExceptHookArgs(
        (*_exc_info(ValueError("boom in a thread")), threading.current_thread())
    )
    caplog.set_level(logging.ERROR, logger="mne_lsl")
    assert _thread_excepthook(args) is None
    assert "boom in a thread" in caplog.text
    assert threading.current_thread().name in caplog.text


def test_thread_excepthook_ignores_system_exit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a thread exiting through 'SystemExit' is not reported."""
    args = threading.ExceptHookArgs(
        (*_exc_info(SystemExit(0)), threading.current_thread())
    )
    caplog.set_level(logging.ERROR, logger="mne_lsl")
    assert _thread_excepthook(args) is None
    assert not caplog.records


def test_exception_policy_keeps_the_event_loop_alive() -> None:
    """Test that an exception raised in a slot neither aborts nor goes unnoticed.

    Run in a subprocess: PyQt6's default policy calls 'qFatal()', which would take the
    whole pytest session down, and pytest-qt owns 'sys.excepthook' inside a test anyway.
    """
    code = (
        "import sys\n"
        "from qtpy.QtCore import QTimer\n"
        "from mne_lsl.viewer._bootstrap import ensure_application\n"
        "from mne_lsl.utils.logs import set_log_level\n"
        "set_log_level('ERROR')\n"
        "app = ensure_application()\n"
        "def raiser():\n"
        "    raise RuntimeError('boom in a slot')\n"
        "QTimer.singleShot(0, raiser)\n"
        "QTimer.singleShot(200, lambda: sys.stdout.write('STILL ALIVE\\n'))\n"
        "QTimer.singleShot(400, app.quit)\n"
        "sys.exit(app.exec())\n"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )
    assert out.returncode == 0, (out.returncode, out.stdout, out.stderr)
    assert "STILL ALIVE" in out.stdout, (out.stdout, out.stderr)
    # the logger writes to stdout, thus the traceback must be visible there.
    assert "boom in a slot" in out.stdout, (out.stdout, out.stderr)


def test_import_mne_lsl_is_qt_free() -> None:
    """Test that 'import mne_lsl' does not import Qt, in a fresh interpreter."""
    code = (
        "import sys; import mne_lsl; "
        "assert not {'qtpy', 'pyqtgraph', 'PyQt6', 'PySide6'} & set(sys.modules), "
        "sorted(set(sys.modules))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
