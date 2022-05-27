"""BSL's logger."""

import logging
import sys

logger = logging.getLogger(__package__.split(".utils", maxsplit=1)[0])
logger.propagate = False  # don't propagate (in case of multiple imports)


def init_logger(verbosity="INFO"):
    """
    Initialize a logger. Assign sys.stdout as a handler of the logger.

    Parameters
    ----------
    verbosity : int | str
        Logger verbosity.
    """
    set_log_level(verbosity)
    add_stream_handler(sys.stdout, verbosity)


def add_stream_handler(stream, verbosity="INFO"):
    """
    Add a handler to the logger. The handler redirects the logger output to
    the stream.

    Parameters
    ----------
    stream : The output stream, e.g. sys.stdout
    verbosity : int | str
        Handler verbosity.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(BSLformatter())
    logger.addHandler(handler)
    set_handler_log_level(verbosity, -1)


def add_file_handler(fname, mode="a", verbosity="INFO"):
    """
    Add a file handler to the logger. The handler saves the logs to file.

    Parameters
    ----------
    fname : str | Path
    mode : str
        Mode in which the file is opened.
    verbosity : int | str
        Handler verbosity.
    """
    handler = logging.FileHandler(fname, mode)
    handler.setFormatter(BSLformatter())
    logger.addHandler(handler)
    set_handler_log_level(verbosity, -1)


def set_handler_log_level(verbosity, handler_id=0):
    """
    Set the log level for a specific handler.
    First handler (ID 0) is always stdout, followed by user-defined handlers.

    Parameters
    ----------
    verbosity : int | str
        Logger verbosity.
    handler_id : int
        ID of the handler among 'logger.handlers'.
    """
    logger.handlers[handler_id].setLevel = verbosity


def set_log_level(verbosity):
    """
    Set the log level for the logger.

    Parameters
    ----------
    verbosity : int | str
        Logger verbosity.
    """
    logger.setLevel(verbosity)


class BSLformatter(logging.Formatter):
    """
    Format string Syntax for BSL.
    """

    # Format string syntax for the different Log levels
    _formatters = dict()
    _formatters[logging.DEBUG] = logging.Formatter(
        fmt="[%(module)s:%(funcName)s:%(lineno)d] %(levelname)s: "
        "%(message)s (%(asctime)s)"
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

    def format(self, record):
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

        return super().format(record)


init_logger()
