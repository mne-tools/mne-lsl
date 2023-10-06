.. include:: ./links.inc

Install
=======

Default install
---------------

``MNE-LSL`` requires Python version ``3.9`` or higher and is available on
`PyPI <project pypi_>`_. It requires ``liblsl`` which will be either fetch from the path
in the environment variable ``MNE_LSL_LIB``, or from the system directories or
downloaded from the release page on compatible platforms.

.. tab-set::

    .. tab-item:: PyPI

        .. code-block:: console

            $ pip install mne-lsl

    .. tab-item:: Source

        .. code-block:: console

            $ pip install git+https://github.com/mne-tools/mne-lsl

Different liblsl version
------------------------

If you prefer to use a different version of `liblsl <lsl lib c++_>`_ than the
automatically downloaded one, or if your platform is not supported, you can build
`liblsl <lsl lib c++_>`_ from source and provide the path to the library in an
environment variable ``MNE_LSL_LIB``.

liblsl and LabRecorder dependencies
-----------------------------------

On Linux, ``liblsl`` requires ``libpugixml-dev`` and ``LabRecorder`` requires
``qt6-base-dev`` and ``freeglut3-dev``.

.. code-block:: console

    $ sudo apt install -y libpugixml-dev qt6-base-dev freeglut3-dev

Qt
--

``MNE-LSL`` requires a Qt binding for the legacy
:class:`~mne_lsl.stream_viewer.StreamViewer` and for the future ``mne_lsl.Viewer``. All
4 Qt bindings, ``PyQt5``, ``PyQt6``, ``PySide2`` and ``PySide6`` are supported thanks to
``qtpy``. It is up to the user to make sure one of the binding is installed in the
environment.

.. warning::

    The legacy :class:`~mne_lsl.stream_viewer.StreamViewer` was developed and tested
    with ``PyQt5`` only.

Optional trigger dependencies
-----------------------------

Parallel port
~~~~~~~~~~~~~

:class:`~mne_lsl.triggers.ParallelPortTrigger` sends trigger (8 bits values) to an
on-board `parallel port`_.

.. tab-set::

    .. tab-item:: Linux

        On Linux systems, the ``pyparallel`` library is required. It can be installed
        either directly or by using the keyword ``triggers`` when installing
        ``MNE-LSL``:

        .. code-block:: console

            $ pip install pyparallel
            $ pip install mne_lsl[triggers]

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
installed either directly or by using the keyword ``triggers`` when installing
``MNE-LSL``:

.. code-block:: console

    $ pip install pyserial
    $ pip install mne_lsl[triggers]

On Linux, the user should be added to the ``dialout`` group which owns the serial port
used:

.. code-block:: console

    $ sudo usermod -aG dialout $USER

Where ``$USER`` is the user name.
