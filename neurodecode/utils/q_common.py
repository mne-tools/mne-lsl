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

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 List/Dict related
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def list2string(vec, fmt, sep=' '):
    """
    Convert a list to string with formatting, separated by sep (default is space).
    Example: fmt= '%.32e', '%.6f', etc.
    """
    return sep.join((fmt % x for x in vec))


def flatten_list(l):
    return list(itertools.chain.from_iterable(l))


def get_index_min(seq):
    """
    Get the index of the minimum item in a list or dict
    """
    if type(seq) == list:
        return min(range(len(seq)), key=seq.__getitem__)
    elif type(seq) == dict:
        return min(seq, key=seq.__getitem__)
    else:
        logger.error('Unsupported input %s' % type(seq))
        return None


def get_index_max(seq):
    """
    Get the index of the maximum item in a list or dict
    """
    if type(seq) == list:
        return max(range(len(seq)), key=seq.__getitem__)
    elif type(seq) == dict:
        return max(seq, key=seq.__getitem__)
    else:
        logger.error('Unsupported input %s' % type(seq))
        return None


def sort_by_value(s, rev=False):
    """
    Sort dictionary or list by value and return a sorted list of keys and values.
    Values must be hashable and unique.
    """
    assert type(s) == dict or type(s) == list, 'Input must be a dictionary or list.'
    if type(s) == list:
        s = dict(enumerate(s))
    s_rev = dict((v, k) for k, v in s.items())
    if Q_VERBOSE > 0 and not len(s_rev) == len(s):
        logger.warning('sort_by_value(): %d identical values' % (len(s.values()) - len(set(s.values())) + 1))
    values = sorted(s_rev, reverse=rev)
    keys = [s_rev[x] for x in values]
    return keys, values


def detect_delim(filename, allowSingleCol=True):
    """
    Automatically find the right delimiter of a file.

    Returns '' if the input file is single column or unknown format.
    If allowSingleCol=False, it will raise an error in the above case.
    """

    temp = open(filename).readline().strip()
    delim = ''
    for d in [',', ' ', '\t']:
        if len(temp.split(d)) > 1:
            delim = d
            break
    else:
        if not allowSingleCol:
            raise Exception('Cannot detect the right delimiter')

    return delim


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 MATLAB
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def matlab(codes):
    """ Execute Matlab snippets """
    exe = 'matlab -nojvm -nodisplay -nosplash -wait -automation -r \"cd %s; %s; exit;\"' % (os.getcwd(), codes)
    # exe= 'matlab -nojvm -nodisplay -nosplash -wait -automation -r \"%s; exit;\"'% codes
    os.system(exe)


def loadmat(filename):
    '''
    Proper mat file loading perserving the correct structure
    https://stackoverflow.com/review/suggested-edits/21667510

    this function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _has_struct(elem):
        """Determine if elem is an array and if any array item is a struct"""
        return isinstance(elem, np.ndarray) and any(isinstance(
                    e, scipy.io.matlab.mio5_params.mat_struct) for e in elem)

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 ETC
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def int2bits(num, nbits=8):
    """ Convert an integer into bits representation. default=8 bits (0-255) """
    return [int(num) >> x & 1 for x in range(nbits - 1, -1, -1)]


def bits2int(bitlist):
    """ Convert a list of bits into an integer """
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out
