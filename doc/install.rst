.. include:: ./links.inc

.. _install:

====================
Install instructions
====================

BSL requires Python version ``3.6`` or higher. BSL is available on `GitHub`_
and on `Pypi <https://pypi.org/project/bsl/>`_.

- **Pypi**: Install BSL using ``pip``

  .. code-block:: console

      $ pip install bsl

- **Conda**: A conda-forge distribution is not yet available.

- **GitHub**: Clone the `main repository <https://github.com/bsl-tools/bsl>`_
  and install with ``setup.py``

  .. code-block:: console

      $ python setup.py install

  For the developer mode, use ``develop`` instead of ``install``.

=====================
Optional dependencies
=====================

BSL installs the following dependencies:

- `numpy`_
- `scipy`_
- `mne`_
- `pyqt5`_
- `pyqtgraph`_

Additional functionalities requires:

- `pyserial`_: for the parallel port trigger using an :ref:`arduino2lpt`.

===========================
Installing pylsl and liblsl
===========================

By default, ``BSL`` is distributed with a recent version of ``pylsl`` and
``liblsl`` that should work on Ubuntu 18.04, Ubuntu 20.04, macOS and
Windows. You can test if your system supports the distributed version by
running the following command in a terminal:

.. code-block:: console

    $ python -c 'from bsl.externals import pylsl'

If you prefer to use a different version, or if the command above raises an
error, `pylsl`_ can be installed in the same environment as ``BSL``. ``BSL``
will automaticaly select `pylsl`_ and will prefer the version installed in the
same environment above the version distributed in ``bsl.externals``.

`pylsl`_ requires a binary library called ``liblsl`` to operate. The binary
library may or may not have been downloaded alongside `pylsl`_. To test if
an install of `pylsl`_ is working, use the following command in a terminal:

.. code-block:: console

    $ python -c 'import pylsl'

If this command did not raise an error, congratulation, you have a working
install of `pylsl`_! However, if an error is raised, please refer to the
instructions below or to the
`LabStreamingLayer Slack <https://labstreaminglayer.slack.com>`_.

Linux
-----

Start by installing ``libpugixml-dev``, a light-weight C++ XML processing
library.

.. code-block:: console

    $ sudo apt install -y libpugixml-dev

Fetch the correct binary from the `liblsl release page`_. If your
distribution is not available, the binary library must be build.
At the time of writing, the binaries version ``1.15.2`` for Ubuntu 18.04
(bionic) and for Ubuntu 20.04 (focal) are available. Install the downloaded
``.deb`` library with:

.. code-block:: console

    $ sudo apt install ./liblsl.deb

macOS
-----

Fetch the correct binary from the `liblsl release page`_ and retrieve the
``.dylib`` binary library. Create an environment variable named ``PYLSL_LIB``
that contains the path to the downloaded binary library.

Alternatively, ``homebrew`` can be used to download and install the binary
library with the command:

.. code-block:: console

    $ brew install labstreaminglayer/tap/lsl

Windows
-------

Fetch the correct binary from the `liblsl release page`_ and retrieve the
``.dll`` binary library. Create an environment variable named ``PYLSL_LIB``
that contains the path to the downloaded binary library.

===================
Installing PsychoPy
===================

The parallel port trigger can either use an on-board parallel port, or an
:ref:`arduino2lpt` connected via USB. The on-board parallel port requires
the ``psychopy.parallel`` module. By default, ``BSL`` is distributed with a
recent version of ``psychopy.parallel`` that should work on most systems.

If you prefer to use a different version, `psychopy`_ can be installed in the
same environment as ``BSL``. ``BSL`` will automaticaly select `psychopy`_ and
will prefer the version installed in the same environment above the version
distributed in ``bsl.externals``.

=====================
Test the installation
=====================

To test the installation, you can run a fake stream with a `~bsl.StreamPlayer`
and display it with a `~bsl.StreamViewer`.

- Download a sample :ref:`bsl.datasets<datasets>`:

  .. code-block:: python

      import bsl
      dataset = bsl.datasets.eeg_resting_state.data_path()
      print (dataset)  # displays the path to the -raw.fif dataset

- Run a `~bsl.StreamPlayer` either from a python console or from terminal using
  the downloaded sample dataset ``resting_state-raw.fif``.

  In a python console:

  .. code-block:: python

      import bsl
      dataset = bsl.datasets.eeg_resting_state.data_path()
      player = StreamPlayer('TestStream', dataset)
      player.start()

  In a terminal, navigate to the folder containing the dataset
  (``~/bsl_data/eeg_sample``):

  .. code-block:: console

      $ bsl_stream_player TestStream resting_state-raw.fif

- Run a `~bsl.StreamViewer` from a different terminal:

  .. code-block:: console

      $ bsl_stream_viewer

The `~bsl.StreamViewer` should load and display:

.. image:: _static/stream_viewer/stream_viewer.gif
   :alt: StreamViewer
   :align: center
