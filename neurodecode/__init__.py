'''
NeuroDecode provides a real-time brain signal decoding framework.

'''

import os
import logging

import neurodecode.colorer
from neurodecode.logger import init_logger

# set loggers
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger  = logging.getLogger('neurodecode')
logger.propagate = False
init_logger(logger, verbose_console='INFO')
