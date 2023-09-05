.. include:: ./links.inc

Install
=======

Default install
---------------

``BSL`` requires Python version ``3.9`` or higher and is available on
`PyPI <project pypi_>`_. It is distributed with a compatible version of
`liblsl <lsl lib c++_>`_.

.. tab-set::

    .. tab-item:: PyPI

        .. code-block:: console

            $ pip install bsl

    .. tab-item:: Source

        .. code-block:: console

            $ pip install git+https://github.com/fcbg-hnp-meeg/bsl

Different liblsl version
------------------------

If you prefer to use a different version of `liblsl <lsl lib c++_>`_, or if your
platform is not supported, you can provide the path to the library in an environment
variable ``LSL_LIB``.

liblsl and LabRecorder dependencies
-----------------------------------

On Linux, ``liblsl`` requires ``libpugixml-dev`` and ``LabRecorder`` requires
``qt6-base-dev`` and ``freeglut3-dev``.

.. code-block:: console

    $ sudo apt install -y libpugixml-dev qt6-base-dev freeglut3-dev

Optional trigger dependencies
-----------------------------

Parallel port
~~~~~~~~~~~~~

:class:`~bsl.triggers.ParallelPortTrigger` sends trigger (8 bits values) to an on-board
`parallel port`_.

.. tab-set::

    .. tab-item:: Linux

        On Linux systems, the ``pyparallel`` library is required. It can be installed
        either directly or by using the keyword ``triggers`` when installing ``BSL``:

        .. code-block:: console

            $ pip install pyparallel
            $ pip install bsl[triggers]

        ``pyparallel`` requires the ``lp`` kernel module to be unloaded. The module can
        be prevented from loading at boot with a file ``blacklist-parallelport.conf``
        containing the single line ``blacklist lp`` in ``/etc/modprobe.d/``.
        Finally, the user should be part of the group owning the parallel port device,
        usually ``lp``.

        .. code-block:: console

            $ ls -l /dev/parport0  # confirm the group owning the parallel port
            $ sudo usermod -aG lp $USER  # add the user to the lp group

        Where ``/dev/parport0`` is the address of the on-board parallel port and
        ``$USER`` is the user name.

    .. tab-item:: Windows

        .. code-block:: console

            On Windows, ``DLPortIO``, ``inpout32`` or ``inpoutx64`` is used.

macOS does not have support for on-board `parallel port`_.

Arduino to parallel port converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Human Neuroscience Platform (FCBG) <fcbg hnp_>`_ has developed an
:ref:`resources/arduino2lpt:Arduino to parallel port (LPT) converter` to replace
on-board parallel ports and to offer hardware triggers to macOS devices. The
``pyserial`` library is required to interface with serial ports (USB). It can be
installed either directly or by using the keyword ``triggers`` when installing ``BSL``:

.. code-block:: console

    $ pip install pyserial
    $ pip install bsl[triggers]

On Linux, the user should be added to the ``dialout`` group which owns the serial port
used:

.. code-block:: console

    $ sudo usermod -aG dialout $USER

Where ``$USER`` is the user name.

Qt
--

``BSL`` requires a Qt binding for the legacy :class:`~bsl.stream_viewer.StreamViewer`
and for the future ``bsl.Viewer``. All 4 Qt bindings, ``PyQt5``, ``PyQt6``, ``PySide2``
and ``PySide6`` are supported thanks to ``qtpy``.

.. warning::

    The legacy :class:`~bsl.stream_viewer.StreamViewer` was developed and tested with
    ``PyQt5`` only.
