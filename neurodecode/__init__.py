"""
NeuroDecode provides a real-time brain signal decoding framework.
"""

from ._version import __version__

import logging

from .logger import init_logger

# set loggers
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger = logging.getLogger('neurodecode')
logger.propagate = False
init_logger(logger, verbosity='INFO')
