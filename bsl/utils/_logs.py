import logging
import sys
from pathlib import Path
from typing import Callable, Optional, TextIO, Union

from ._checks import _check_verbose
from ._docs import fill_doc

logger = logging.getLogger(__package__.split(".utils", maxsplit=1)[0])
logger.propagate = False  # don't propagate (in case of multiple imports)


@fill_doc
def _init_logger(verbose: Optional[Union[bool, str, int]] = None) -> None:
    """Initialize a logger.

    Assign sys.stdout as a handler of the logger.

    Parameters
    ----------
    %(verbose)s
    """
    set_log_level(verbose)
    add_stream_handler(sys.stdout, verbose)


@fill_doc
def add_stream_handler(
    stream: TextIO, verbose: Optional[Union[bool, str, int]] = None
) -> None:
    """Add a stream handler to the logger.

    Parameters
    ----------
    stream : TextIO
        The output stream, e.g. ``sys.stdout``.
    %(verbose)s
    """
    verbose = _check_verbose(verbose)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(LoggerFormatter())
    logger.addHandler(handler)
    set_handler_log_level(-1, verbose)


@fill_doc
def add_file_handler(
    fname: Union[str, Path],
    mode: str = "a",
    encoding: Optional[str] = None,
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
    handler.setFormatter(LoggerFormatter())
    logger.addHandler(handler)
    set_handler_log_level(-1, verbose)


@fill_doc
def set_handler_log_level(
    handler_id: int, verbose: Union[bool, str, int, None]
) -> None:
    """Set the log level for a specific handler.

    First handler (ID 0) is always ``sys.stdout``, followed by user-defined
    handlers.

    Parameters
    ----------
    handler_id : int
        ID of the handler among ``logger.handlers``.
    %(verbose)s
    """
    verbose = _check_verbose(verbose)
    logger.handlers[handler_id].setLevel = verbose


@fill_doc
def set_log_level(verbose: Union[bool, str, int, None]) -> None:
    """Set the log level for the logger.

    Parameters
    ----------
    %(verbose)s
    """
    verbose = _check_verbose(verbose)
    logger.setLevel(verbose)


class LoggerFormatter(logging.Formatter):
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

    def wrapper(*args, **kwargs):
        if "verbose" in kwargs:
            set_log_level(kwargs["verbose"])
        return f(*args, **kwargs)

    return wrapper


_init_logger()
