'''
NeuroDecode provides a real-time brain signal decoding framework.

'''

import os
import logging

import neurodecode.colorer
from neurodecode.logger import init_logger

# set loggers
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('neurodecode').propagate = False
init_logger(logging.getLogger('neurodecode'))
