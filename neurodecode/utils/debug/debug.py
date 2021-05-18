import os
import sys
import pdb
import code
import inspect
import multiprocessing as mp

from neurodecode import logger


#----------------------------------------------------------------------
def auto_debug():
    """
    Triggers debugging mode automatically when AssertionError is raised

    Snippet from: stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
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

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
def run_multi(cmd_list, cores=0, quiet=False):
    """
    Run multiple command in separate process.
    
    Logging tip: "command args > log.txt 2>&1"

    Parameters
    ----------
    cmd_list : list
        The list of commands to run
    cores : int
        The number of cores to use (use all cores if 0)
    quiet : bool
        If True, display the command into the logger
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
