.. include:: ./links.inc

.. _install:

====================
Install instructions
====================

BSL requires Python version ``3.6`` or higher. BSL is available on `GitHub`_
and on `Pypi <https://pypi.org/project/bsl/>`_.

- **GitHub**: Clone the `main repository <https://github.com/bsl-tools/bsl>`_
  and install with ``setup.py``

  .. code-block:: console

      $ python setup.py install

  For the developer mode, use ``develop`` instead of ``install``.

- **Pypi**: Install BSL using ``pip``

  .. code-block:: console

      $ pip install bsl

- **Conda**: A conda-forge distribution is not yet available.

=====================
Optional dependencies
=====================

BSL installs the following dependencies:

- `numpy`_
- `scipy`_
- `mne`_
- `pylsl`_
- `pyqt5`_
- `pyqtgraph`_

Additional functionalities requires:

- `psychopy`_: for the parallel port triggers.
- `pyserial`_: for the parallel port trigger using an :ref:`arduino2lpt`.
- `vispy <https://vispy.org/>`_: for the `~bsl.StreamViewer` vispy backend.

===========================
Installing pylsl and liblsl
===========================

`pylsl`_ requires a binary library, called ``liblsl`` to operate. But the
binary library might not been downloaded alongside `pylsl`_.

To test if `pylsl`_ is working, try the following command in a terminal:

.. code-block:: console

    $ python -c 'import pylsl'

If it didn't raise an error, congratulation, you have a working setup! However,
if an error is raised, please refer to the instructions below or to the
`LabStreamingLayer Slack <https://labstreaminglayer.slack.com>`_.

Linux
-----

Start by installing ``libpugixml-dev``, a light-weight C++ XML processing
library.

.. code-block:: console

    $ sudo apt install -y libpugixml-dev

Fetch the correct binary from the `liblsl release page`_. If your
distribution is not available, the binary library must be build.
At the time of writing, the binary version ``1.15.2`` for Ubuntu 18.04
(bionic) and for Ubuntu 20.04 (focal) are available.
Create an environment variable named ``PYLSL_LIB`` that contains the path
to the downloaded binary library.

macOS
-----

Fetch the correct binary from the `liblsl release page`_. If your
distribution is not available, the binary library must be build.
Create an environment variable named ``PYLSL_LIB`` that contains the path
to the downloaded binary library.

Alternatively, ``homebrew`` can be used to download and set the binary
library with the command:

.. code-block:: console

    $ brew install labstreaminglayer/tap/lsl


Windows
-------

Fetch the correct binary from the `liblsl release page`_. If your
distribution is not available, the binary library must be build.
Create an environment variable named ``PYLSL_LIB`` that contains the path
to the downloaded binary library.

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

  In terminal, navigate to the folder containing the dataset
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
