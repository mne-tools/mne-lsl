from __future__ import print_function, division

"""
Python utilities

Kyuhwa Lee (kyu.lee@epfl.ch)
Swiss Federal Institute of Technology of Lausanne (EPFL)

"""

# set Q_VERBOSE= 0 to make it silent. 1:verbose, 2:extra verbose
Q_VERBOSE = 0


import os
import sys
import pdb
import code
import scipy
import inspect
import itertools
import numpy as np
import multiprocessing as mp

from neurodecode import logger

# pickle
try:
    import cPickle as pickle  # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle  # Python 3 (C version is the default)


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Debugging
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def auto_debug():
    """
    Triggers debugging mode automatically when AssertionError is raised

    Snippet from:
      stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
    """
    def debug_info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type == KeyboardInterrupt:
            # interactive mode or no tty-like device
            sys.__excepthook__(type, value, tb)
        else:
            # non-interactive mode
            logger.exception()
            pdb.pm()
    sys.excepthook = debug_info


# enter interactive shell within the caller's scope
def shell():
    """
    Enter interactive shell within the caller's scope
    """
    logger.info('*** Entering interactive shell. Ctrl+D to return. ***')
    stack = inspect.stack()
    try:  # globals are first loaded, then overwritten by locals
        globals_ = {}
        globals_.update({key:value for key, value in stack[1][0].f_globals.items()})
        globals_.update({key:value for key, value in stack[1][0].f_locals.items()})
    finally:
        del stack
    code.InteractiveConsole(globals_).interact()


def run_multi(cmd_list, cores=0, quiet=False):
    """
    Input
    -----
    cmd_list: list of commands just like when you type on bash
    cores: number of cores to use (use all cores if 0)
    Logging tip: "command args > log.txt 2>&1"
    """
    if cores == 0: cores = mp.cpu_count()
    pool = mp.Pool(cores)
    processes = []
    for cmd in cmd_list:
        if not quiet:
            logger.info(cmd)
        processes.append(pool.apply_async(os.system, [cmd]))
    for proc in processes:
        proc.get()
    pool.close()
    pool.join()
