from __future__ import print_function

"""
Convenient start-up Python script.

To automatically load this script at Python startup, add this file's path
to the Python enviornment variable PYTHONSTARTUP.

"""

print('\nLoading startup modules... ', end='')
import pycnbi_config
import q_common as qc
import numpy as np

print('Done.')
