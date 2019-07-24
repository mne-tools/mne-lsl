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

# log level options provided by pycnbi
LOG_LEVELS = {
    'DEBUG':logging.DEBUG,
    'INFO':logging.INFO,
    'INFO_GREEN':22,
    'INFO_BLUE':24,
    'INFO_YELLOW':26,
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
        if record.levelno == LOG_LEVELS['DEBUG']:
            self._fmt = self.fmt_debug
        elif record.levelno == LOG_LEVELS['INFO']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['INFO_GREEN']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['INFO_BLUE']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['INFO_YELLOW']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['WARNING']:
            self._fmt = self.fmt_warning
        elif record.levelno >= LOG_LEVELS['ERROR']:
            self._fmt = self.fmt_error
        self._style = logging.PercentStyle(self._fmt)
        return logging.Formatter.format(self, record)

def init_logger(verbose_console='INFO'):
    '''
    Add the first logger as sys.stdout. Handler will be added only once.
    '''
    if not logger.hasHandlers():
        add_logger_handler(sys.stdout, verbosity=verbose_console)
    
    '''
    TODO: add file handler
    # file logger handler
    f_handler = logging.FileHandler('pycnbi.log', mode='a')
    f_handler.setLevel(loglevels[verbose_file])
    f_format = logging.Formatter('%(levelname)s %(asctime)s %(funcName)s:%(lineno)d: %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    '''

def add_logger_handler(stream, verbosity='INFO'):
    # add custom log levels
    logging.addLevelName(LOG_LEVELS['INFO_GREEN'], 'INFO_GREEN')
    def __log_info_green(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_LEVELS['INFO_GREEN']):
            self._log(LOG_LEVELS['INFO_GREEN'], message, args, **kwargs)
    logging.Logger.info_green = __log_info_green

    logging.addLevelName(LOG_LEVELS['INFO_BLUE'], 'INFO_BLUE')
    def __log_info_blue(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_LEVELS['INFO_BLUE']):
            self._log(LOG_LEVELS['INFO_BLUE'], message, args, **kwargs)
    logging.Logger.info_blue = __log_info_blue

    logging.addLevelName(LOG_LEVELS['INFO_YELLOW'], 'INFO_YELLOW')
    def __log_info_yellow(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_LEVELS['INFO_YELLOW']):
            self._log(LOG_LEVELS['INFO_YELLOW'], message, args, **kwargs)
    logging.Logger.info_yellow = __log_info_yellow

    # console logger handler
    c_handler = logging.StreamHandler(stream)
    c_handler.setFormatter(PycnbiFormatter())
    logger.addHandler(c_handler)

    # minimum possible level of all handlers
    logger.setLevel(logging.DEBUG)

    logger.handlers[-1].level = LOG_LEVELS[verbosity]
    set_log_level(verbosity)
    return logger

def set_log_level(verbosity, handler_id=0):
    '''
    hander ID 0 is always stdout, followed by user-defined handlers.
    '''
    logger.handlers[handler_id].level = LOG_LEVELS[verbosity]

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
