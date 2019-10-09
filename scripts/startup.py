from __future__ import print_function

"""
Convenient start-up Python script.

To automatically load this script at Python startup, add this file's path
to the Python enviornment variable PYTHONSTARTUP.

"""

print('\nLoading startup modules... ', end='')

import mne
import numpy as np
import neurodecode.utils.neurodecode_utils as pu
import neurodecode.utils.q_common as qc

print('Done.')
