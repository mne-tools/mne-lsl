.. include:: ./links.inc

.. _install:

Install
=======

Default install
---------------

``BSL`` requires Python version ``3.8`` or higher and is available on
`PyPI <project pypi_>`_. It is distributed with a compatible version of
`liblsl <lsl lib c++_>`_.

.. tab-set::

    .. tab-item:: PyPI

        .. code-block:: console

            $ pip install bsl

    .. tab-item:: conda-forge

        Not yet available.

    .. tab-item:: Source

        .. code-block:: console

            $ pip install git+https://github.com/fcbg-hnp-meeg/bsl

Optional dependencies
---------------------

Parallel port
^^^^^^^^^^^^^

.. _parallel port: https://en.wikipedia.org/wiki/Parallel_port

`~bsl.triggers.ParallelPortTrigger` sends trigger (8 bits values) to a `parallel port`_.
On Linux, the ``pyparallel`` library is required. If an
:ref:`arduino_lpt:Arduino to parallel port (LPT) converter` is used, the ``pyserial``
library is required. Both can be installed using the extra-key ``triggers``:

.. code-block:: console

    $ pip install bsl[triggers]

On Linux, the user must have access to the parallel port. For instance, if an onboard
parallel port at the address ``/dev/parport0`` is used, you can check the group owning
the device with:

.. code-block:: console

    $ ls -l /dev/parport0

Usually, the group is ``lp``. The user should be added to this group:

.. code-block:: console

    $ sudo usermod -aG lp $USER

Moreover, ``pyparallel`` requires the ``lp`` kernel module to be unloaded. This can be
done at boot with a ``blacklist-parallelport.conf`` file containing the line
``blacklist lp`` in ``/etc/modprobe.d/``.

If an :ref:`arduino_lpt:Arduino to parallel port (LPT) converter` is used, the user
should be added to the ``dialout`` group which owns the serial port used:

.. code-block:: console

    $ sudo usermod -aG dialout $USER

Qt
^^

At the moment, ``BSL`` requires ``PyQt5``. Future versions will support other Qt
bindings via ``qtpy``. On Linux based distribution, ``PyQt5`` requires system libraries:

.. code-block:: console

    $ sudo apt install -y qt5-default  # Ubuntu 20.04 LTS
    $ sudo apt install -y qtbase5-dev qt5-qmake  # Ubuntu 22.04 LTS

Advance install
---------------

By default, ``BSL`` is distributed with a recent version of ``liblsl`` that should work
on Ubuntu-based distribution, macOS and Windows. If your OS is not compatible with the
distributed version or if you want to use a specific ``liblsl``, provide the path to the
library in an environment variable ``LSL_LIB``.

Troubleshooting
---------------

On Linux, ``liblsl`` requires ``libpugixml-dev`` and ``LabRecorder`` requires
``qt6-base-dev`` and ``freeglut3-dev``.

.. code-block:: console

    $ sudo apt install -y libpugixml-dev qt6-base-dev freeglut3-dev

On macOS, ``homebrew`` can be used to download and install ``liblsl``:

.. code-block:: console

    $ brew install labstreaminglayer/tap/lsl

To test the installation, you can run a fake stream with a `~bsl.StreamPlayer` and
display it with a `~bsl.StreamViewer`.

- Download a sample :ref:`bsl.datasets<datasets>`:

  .. code-block:: python

      import bsl
      dataset = bsl.datasets.eeg_resting_state.data_path()
      print (dataset)  # displays the path to the -raw.fif dataset

- Run a `~bsl.StreamPlayer` either from a python console:

  .. code-block:: python

      import bsl
      dataset = bsl.datasets.eeg_resting_state.data_path()
      player = StreamPlayer('TestStream', dataset)
      player.start()

  Or from a terminal in the folder containing the dataset (``~/bsl_data/eeg_sample``):

  .. code-block:: console

      $ bsl_stream_player TestStream resting_state-raw.fif

- Run a `~bsl.StreamViewer` from a different terminal:

  .. code-block:: console

      $ bsl_stream_viewer

  The `~bsl.StreamViewer` should load and display:

  .. image:: _static/stream_viewer/stream_viewer.gif
      :alt: StreamViewer
      :align: center
