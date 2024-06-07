import logging
from logging import Logger
from pathlib import Path
from typing import Callable

from _typeshed import Incomplete

from ._checks import check_verbose as check_verbose
from ._docs import fill_doc as fill_doc
from ._fixes import _WrapStdOut as _WrapStdOut

_PACKAGE: str

def _init_logger(*, verbose: bool | str | int | None = ...) -> Logger:
    """Initialize a logger.

    Assigns sys.stdout as the first handler of the logger.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between
        ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
        If None is provided, the verbosity is set to the currently set logger's level.
        If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
        to ``"INFO"`` for True.

    Returns
    -------
    logger : Logger
        The initialized logger.
    """

def add_file_handler(
    fname: str | Path,
    mode: str = "a",
    encoding: str | None = None,
    *,
    verbose: bool | str | int | None = None,
) -> None:
    """Add a file handler to the logger.

    Parameters
    ----------
    fname : str | Path
        Path to the file where the logging output is saved.
    mode : str
        Mode in which the file is opened.
    encoding : str | None
        If not None, encoding used to open the file.
    verbose : int | str | bool | None
        Sets the verbosity level of the file handler. The verbosity increases gradually
        between ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
        If a bool is provided, the verbosity is set to ``"WARNING"`` for False and to
        ``"INFO"`` for True. If None is provided, the verbosity of the logger is used.
    """

def set_log_level(verbose: bool | str | int | None) -> None:
    """Set the log level for the logger.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between
        ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
        If None is provided, the verbosity is set to the currently set logger's level.
        If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
        to ``"INFO"`` for True.
    """

class _LoggerFormatter(logging.Formatter):
    """Format string Syntax."""

    _formatters: Incomplete

    def __init__(self) -> None: ...
    def format(self, record: logging.LogRecord):
        """
        Format the received log record.

        Parameters
        ----------
        record : logging.LogRecord
        """

def verbose(f: Callable) -> Callable:
    """Set the verbose for the function call from the kwargs.

    Parameters
    ----------
    f : callable
        The function with a verbose argument.

    Returns
    -------
    f : callable
        The function.
    """

class _use_log_level:
    """Context manager to change the logging level temporary.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between
        ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
        If None is provided, the verbosity is set to the currently set logger's level.
        If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
        to ``"INFO"`` for True.
    """

    _logger: Incomplete
    _old_level: Incomplete
    _level: Incomplete

    def __init__(
        self, verbose: bool | str | int | None = None, logger_obj: Logger | None = None
    ) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...

def warn(
    message: str,
    category: Warning = ...,
    module: str = ...,
    ignore_namespaces: tuple[str, ...] | list[str] = ...,
) -> None:
    """Emit a warning with trace outside the requested namespace.

    This function takes arguments like :func:`warnings.warn`, and sends messages
    using both :func:`warnings.warn` and :func:`logging.warn`. Warnings can be
    generated deep within nested function calls. In order to provide a
    more helpful warning, this function traverses the stack until it
    reaches a frame outside the ignored namespace that caused the error.

    This function is inspired from the MNE-Python package and behaves as a smart
    'stacklevel' argument.

    Parameters
    ----------
    message : str
        Warning message.
    category : instance of Warning
        The warning class. Defaults to ``RuntimeWarning``.
    module : str
        The name of the module emitting the warning.
    ignore_namespaces : list of str | tuple of str
        Namespaces to ignore when traversing the stack.
    """

logger: Incomplete
