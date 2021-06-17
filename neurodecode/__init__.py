"""
NeuroDecode provides a real-time brain signal decoding framework.
"""
import logging

from .logger import init_logger

# set loggers
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger = logging.getLogger('neurodecode')
logger.propagate = False
init_logger(logger, verbosity='INFO')
