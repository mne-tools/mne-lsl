#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
"""

import sys

from .... import logger

# To make life easier, only try drivers which have a hope in heck of working.
# Because hasattr() in connection to windll ends up in an OSError trying to
# load 32bit drivers in a 64bit environment, different drivers defined in
# the dictionary 'drivers' are tested.

if sys.platform.startswith('linux'):
    from ._linux import PParallelLinux
    ParallelPort = PParallelLinux
elif sys.platform == 'win32':
    drivers = dict(inpout32=('_inpout', 'PParallelInpOut'),
                   inpoutx64=('_inpout', 'PParallelInpOut'),
                   dlportio=('_dlportio', 'PParallelDLPortIO'))
    from ctypes import windll
    from importlib import import_module
    for key, val in drivers.items():
        driver_name, class_name = val
        try:
            hasattr(windll, key)
            ParallelPort = getattr(import_module('.'+driver_name, __name__),
                                   class_name)
            break
        except (OSError, KeyError, NameError):
            ParallelPort = None
            continue
    if ParallelPort is None:
        logger.warning("psychopy.parallel has been imported but no "
                       "parallel port driver found. Install either "
                       "inpout32, inpoutx64 or dlportio")
else:
    logger.warning("psychopy.parallel has been imported on a Mac "
                   "(which doesn't have a parallel port?)")

    # macOS doesn't have a parallel port but write the class for doc purps
    class ParallelPort:
        """Class for read/write access to the parallel port on Windows & Linux

        Usage::

            from psychopy import parallel
            port = parallel.ParallelPort(address=0x0378)

            port.setData(4)
            port.readPin(2)
            port.setPin(2, 1)
        """

        def __init__(self, address):
            """This is just a dummy constructor to avoid errors
            when the parallel port cannot be initiated
            """
            msg = ("psychopy.parallel has been imported but (1) no parallel "
                   "port driver could be found or accessed on Windows or "
                   "(2) PsychoPy is run on a Mac (without parallel-port "
                   "support for now)")
            logger.warning(msg)

        def setData(self, data):
            """Set the data to be presented on the parallel port (one ubyte).
            Alternatively you can set the value of each pin (data pins are
            pins 2-9 inclusive) using :func:`~psychopy.parallel.setPin`

            Examples::

                from psychopy import parallel
                port = parallel.ParallelPort(address=0x0378)

                port.setData(0)  # sets all pins low
                port.setData(255)  # sets all pins high
                port.setData(2)  # sets just pin 3 high (pin2 = bit0)
                port.setData(3)  # sets just pins 2 and 3 high

            You can also convert base 2 to int easily in python::

                port.setData( int("00000011", 2) )  # pins 2 and 3 high
                port.setData( int("00000101", 2) )  # pins 2 and 4 high
            """
            sys.stdout.flush()
            raise NotImplementedError("Parallel ports don't work on a Mac")

        def readData(self):
            """Return the value currently set on the data pins (2-9)
            """
            raise NotImplementedError("Parallel ports don't work on a Mac")

        def readPin(self, pinNumber):
            """Determine whether a desired (input) pin is high(1) or low(0).

            Pins 2-13 and 15 are currently read here
            """
            raise NotImplementedError("Parallel ports don't work on a Mac")
