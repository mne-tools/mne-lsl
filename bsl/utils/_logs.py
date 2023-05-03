import logging
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union

from ._checks import _check_verbose
from ._docs import fill_doc
from ._fixes import _WrapStdOut


@fill_doc
def _init_logger(*, verbose: Optional[Union[bool, str, int]] = None) -> logging.Logger:
    """Initialize a logger.

    Assigns sys.stdout as the first handler of the logger.

    Parameters
    ----------
    %(verbose)s

    Returns
    -------
    logger : Logger
        The initialized logger.
    """
    # create logger
    verbose = _check_verbose(verbose)
    logger = logging.getLogger(__package__.split(".utils", maxsplit=1)[0])
    logger.propagate = False
    logger.setLevel(verbose)

    # add the main handler
    handler = logging.StreamHandler(_WrapStdOut())
    handler.setFormatter(_LoggerFormatter())
    logger.addHandler(handler)

    return logger


@fill_doc
def add_file_handler(
    fname: Union[str, Path],
    mode: str = "a",
    encoding: Optional[str] = None,
    *,
    verbose: Optional[Union[bool, str, int]] = None,
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
    %(verbose)s
    """
    verbose = _check_verbose(verbose)
    handler = logging.FileHandler(fname, mode, encoding)
    handler.setFormatter(_LoggerFormatter())
    handler.setLevel(verbose)
    logger.addHandler(handler)


@fill_doc
def set_log_level(verbose: Optional[Union[bool, str, int]]) -> None:
    """Set the log level for the logger and the first handler ``sys.stdout``.

    Parameters
    ----------
    %(verbose)s
    """
    verbose = _check_verbose(verbose)
    logger.setLevel(verbose)


class _LoggerFormatter(logging.Formatter):
    """Format string Syntax."""

    # Format string syntax for the different Log levels
    _formatters = dict()
    _formatters[logging.DEBUG] = logging.Formatter(
        fmt="[%(module)s:%(funcName)s:%(lineno)d] %(levelname)s: %(message)s "
        "(%(asctime)s)"
    )
    _formatters[logging.INFO] = logging.Formatter(
        fmt="[%(module)s.%(funcName)s] %(levelname)s: %(message)s"
    )
    _formatters[logging.WARNING] = logging.Formatter(
        fmt="[%(module)s.%(funcName)s] %(levelname)s: %(message)s"
    )
    _formatters[logging.ERROR] = logging.Formatter(
        fmt="[%(module)s:%(funcName)s:%(lineno)d] %(levelname)s: %(message)s"
    )

    def __init__(self):
        super().__init__(fmt="%(levelname): %(message)s")

    def format(self, record: logging.LogRecord):
        """
        Format the received log record.

        Parameters
        ----------
        record : logging.LogRecord
        """
        if record.levelno <= logging.DEBUG:
            return self._formatters[logging.DEBUG].format(record)
        elif record.levelno <= logging.INFO:
            return self._formatters[logging.INFO].format(record)
        elif record.levelno <= logging.WARNING:
            return self._formatters[logging.WARNING].format(record)
        else:
            return self._formatters[logging.ERROR].format(record)


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

    @wraps(f)
    def wrapper(*args, **kwargs):
        if "verbose" in kwargs:
            with _use_log_level(kwargs["verbose"]):
                return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)

    return wrapper


@fill_doc
class _use_log_level:
    """Context manager to change the logging level temporary.

    Parameters
    ----------
    %(verbose)s
    """

    def __init__(self, verbose: Union[bool, str, int, None] = None):
        self._old_level = logger.level
        self._level = verbose

    def __enter__(self):
        set_log_level(self._level)

    def __exit__(self, *args):
        set_log_level(self._old_level)


logger = _init_logger(verbose="WARNING")  # equivalent to verbose=None
