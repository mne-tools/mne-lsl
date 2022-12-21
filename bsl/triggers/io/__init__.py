"""
This module provides read / write access to the parallel port for
Linux or Windows.

The :class:`~psychopy.parallel.Parallel` class described below will
attempt to load whichever parallel port driver is first found on your
system and should suffice in most instances. If you need to use a specific
driver then, instead of using :class:`~psychopy.parallel.ParallelPort`
shown below you can use one of the following as drop-in replacements,
forcing the use of a specific driver:

    - `psychopy.parallel.PParallelInpOut`
    - `psychopy.parallel.PParallelDLPortIO`
    - `psychopy.parallel.PParallelLinux`

Either way, each instance of the class can provide access to a different
parallel port.

This code snippet is inspired from the parallel module of PsychoPy.
"""

import sys

# To make life easier, only try drivers which have a hope in heck of working.
# Because hasattr() in connection to windll ends up in an OSError trying to
# load 32bit drivers in a 64bit environment, different drivers defined in
# the dictionary 'drivers' are tested.

if sys.platform.startswith("linux"):
    from ._linux import PParallelLinux

    ParallelPort = PParallelLinux
elif sys.platform == "win32":
    drivers = dict(
        inpout32=("_inpout", "PParallelInpOut"),
        inpoutx64=("_inpout", "PParallelInpOut"),
        dlportio=("_dlportio", "PParallelDLPortIO"),
    )
    from ctypes import windll
    from importlib import import_module

    for key, val in drivers.items():
        driver_name, class_name = val
        try:
            hasattr(windll, key)
            ParallelPort = getattr(
                import_module("." + driver_name, __name__), class_name
            )
            break
        except (OSError, KeyError, NameError):
            ParallelPort = None
            continue
else:
    ParallelPort = None
