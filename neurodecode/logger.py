import sys
import logging

# log level options provided by neurodecode
LOG_LEVELS = {
    'DEBUG':logging.DEBUG,
    'INFO':logging.INFO,
    'INFO_GREEN':22,
    'INFO_BLUE':24,
    'INFO_YELLOW':26,
    'WARNING':logging.WARNING,
    'ERROR':logging.ERROR
}

def init_logger(logger, verbose_console='INFO'):
    '''
    Initialize a logger. 

    Assign sys.stdout as a handler of the logger.

    logger = the logger
    verbose_console = logger verbosity
    '''
    if not logger.hasHandlers():
        logger.setLevel(verbose_console)
        add_logger_handler(logger, sys.stdout, verbosity=verbose_console)

    '''
    TODO: add file handler
    # file logger handler
    f_handler = logging.FileHandler('neurodecode.log', mode='a')
    f_handler.setLevel(loglevels[verbose_file])
    f_format = logging.Formatter('%(levelname)s %(asctime)s %(funcName)s:%(lineno)d: %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
   '''

def add_logger_handler(logger, stream, verbosity='INFO'):
    '''
    Add a handler to the logger.

    The handler redirects the logger output to the stream.

    logger = The logger
    stream = The output stream
    '''

    add_customs_log_levels()

    c_handler = logging.StreamHandler(stream)
    c_handler.setFormatter(neurodecodeFormatter())
    logger.addHandler(c_handler)

    set_log_level(logger, verbosity, -1)
    
    return logger

def add_customs_log_levels():
    '''
    Add custom levels to the logger

    Three additional neurodecode-specific log levels (INFO_GREEN, INFO_BLUE and INFO_YELLOW)
    are added.
    '''
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

def set_log_level(logger, verbosity, handler_id=0):
    '''
    Set the log level for a specific handler

    First handler (ID 0) is always stdout, followed by user-defined handlers.

    handler_id = handler ID
    '''
    logger.handlers[handler_id].level = LOG_LEVELS[verbosity]



class neurodecodeFormatter(logging.Formatter):
    '''
    Format string Syntax for Neurodecode

    TODO: support a process-safe file logging
    '''
    
    # Format string syntax for the different Log levels
    fmt_debug = "[%(module)s:%(funcName)s:%(lineno)d] DEBUG: %(message)s (%(asctime)s)"
    fmt_info = "[%(module)s.%(funcName)s] %(message)s"
    fmt_warning = "[%(module)s.%(funcName)s] WARNING: %(message)s"
    fmt_error = "[%(module)s:%(funcName)s:%(lineno)d] ERROR: %(message)s"

    def __init__(self, fmt="%(levelno)s: %(message)s"):
        '''
        Initialize the neurodecode formatter

        fmt = format string syntax
        '''
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        '''
        Format the received log record 

        record = log record to format
        '''
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