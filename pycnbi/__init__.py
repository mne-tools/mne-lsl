'''
Initialize PyCNBI logger and other settings

TODO: support a process-safe file logging

Kyuhwa Lee
Swiss Federal Institute of Technology Lausanne (EPFL)

'''

import os
import sys
import logging
import pycnbi.colorer
from pycnbi.utils import q_common as qc

# log level options
LOG_LEVELS = {
    'DEBUG':logging.DEBUG,
    'INFO':logging.INFO,
    'WARNING':logging.WARNING,
    'ERROR':logging.ERROR
}

class PycnbiFormatter(logging.Formatter):
    fmt_debug = "[%(module)s:%(funcName)s:%(lineno)d] DEBUG: %(message)s (%(asctime)s)"
    fmt_info = "[%(module)s.%(funcName)s] %(message)s"
    fmt_warning = "[%(module)s.%(funcName)s] WARNING: %(message)s"
    fmt_error = "[%(module)s:%(funcName)s:%(lineno)d] ERROR: %(message)s"

    def __init__(self, fmt="%(levelno)s: %(message)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.DEBUG:
            self._fmt = self.fmt_debug
        elif record.levelno == logging.INFO:
            self._fmt = self.fmt_info
        elif record.levelno == logging.WARNING:
            self._fmt = self.fmt_warning
        elif record.levelno >= logging.ERROR:
            self._fmt = self.fmt_error
        self._style = logging.PercentStyle(self._fmt)
        return logging.Formatter.format(self, record)

def init_logger(verbose_console='INFO', verbose_file=None):
    if not logger.hasHandlers():
        # console logger handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(PycnbiFormatter())
        logger.addHandler(c_handler)

        '''
        # file logger handler
        f_handler = logging.FileHandler('pycnbi.log', mode='a')
        f_handler.setLevel(loglevels[verbose_file])
        f_format = logging.Formatter('%(levelname)s %(asctime)s %(funcName)s:%(lineno)d: %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
        '''

        # minimum possible level of all handlers
        logger.setLevel(logging.DEBUG)

    set_log_level(verbose_console, verbose_file)
    return logger

def set_log_level(verbose_console, verbose_file=None):
    logger.handlers[0].level = LOG_LEVELS[verbose_console]
    if verbose_file is not None:
        # logger.handlers[1].level = verbose_file
        raise NotImplementedError("Sorry, file logging is not supported yet because I don't know how to use with multiprcocessing.")


# init scripts
ROOT = qc.parse_path(os.path.realpath(__file__)).dir
for d in qc.get_dir_list(ROOT):
    if os.path.exists('%s/__init__.py' % d):
        exe_package = 'import pycnbi.%s' % d.replace(ROOT + '/', '')
        exec(exe_package)

# set loggers
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger = logging.getLogger('pycnbi')
logger.propagate = False
init_logger()
