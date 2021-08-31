.. _install:

====================
Install instructions
====================

BSL requires Python version 3.6 or higher. BSL is available on GitHub and on
Pypi.

- GitHub: Clone the main repository and install with ``setup.py``:

  .. code-block:: console

      $ python setup.py install

  For the developer mode, use ``develop`` instead of ``install``.

- Pypi: Install BSL using ``pip``:

  .. code-block:: console

      $ pip install bsl

- Conda: Not yet available.

=====================
Optional dependencies
=====================

BSL installs the following dependencies:

- numpy
- scipy
- mne
- pylsl
- pyqt5
- pyqtgraph

Additionnal functionalities requires:

- pyserial: for the trigger using an LPT to Arduino converter
- vispy: for the Stream Viewer vispy backend
