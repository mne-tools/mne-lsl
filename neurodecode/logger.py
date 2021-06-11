"""
Neurodecode's logger.
TODO: Add file handler.
"""
import sys
import logging

# log level options provided by neurodecode
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR
}


def init_logger(logger, verbosity='INFO'):
    """
    Initialize a logger. Assign sys.stdout as a handler of the logger.

    Parameters
    ----------
    logger : logging.Logger
    verbosity : str
        The logger verbosity. Supported: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    """
    if not logger.hasHandlers():
        logger.setLevel(verbosity)
        add_logger_handler(logger, sys.stdout, verbosity=verbosity)


def add_logger_handler(logger, stream, verbosity='INFO'):
    """
    Add a handler to the logger. The handler redirects the logger output to
    the stream.

    Parameters
    ----------
    logger : logging.Logger
    stream : The output stream, e.g. sys.stdout
    """
    c_handler = logging.StreamHandler(stream)
    c_handler.setFormatter(neurodecodeFormatter())
    logger.addHandler(c_handler)

    set_log_level(logger, verbosity, -1)

    return logger


def set_log_level(logger, verbosity, handler_id=0):
    """
    Set the log level for a specific handler.
    First handler (ID 0) is always stdout, followed by user-defined handlers.

    Parameters
    ----------
    logger : logging.Logger
    verbosity : str
        The logger verbosity. Supported: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    handler_id : int
        The ID of the handler among 'logger.handlers'.
    """
    logger.handlers[handler_id].level = LOG_LEVELS[verbosity]


class neurodecodeFormatter(logging.Formatter):
    """
    Format string Syntax for Neurodecode.

    Parameters
    ----------
    fmt : str
        Format string syntax. The default is '%(levelno)s: %(message)s'
    """

    # Format string syntax for the different Log levels
    fmt_debug = "[%(module)s:%(funcName)s:%(lineno)d] DEBUG: %(message)s (%(asctime)s)"
    fmt_info = "[%(module)s.%(funcName)s] %(message)s"
    fmt_warning = "[%(module)s.%(funcName)s] WARNING: %(message)s"
    fmt_error = "[%(module)s:%(funcName)s:%(lineno)d] ERROR: %(message)s"

    def __init__(self, fmt='%(levelno)s: %(message)s'):
        super().__init__(fmt)

    def format(self, record):
        """
        Format the received log record.

        Parameters
        ----------
        record : logging.LogRecord
        """
        if record.levelno == LOG_LEVELS['DEBUG']:
            self._fmt = self.fmt_debug
        elif record.levelno == LOG_LEVELS['INFO']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['WARNING']:
            self._fmt = self.fmt_warning
        elif record.levelno >= LOG_LEVELS['ERROR']:
            self._fmt = self.fmt_error

        self._style = logging.PercentStyle(self._fmt)

        return logging.Formatter.format(self, record)
