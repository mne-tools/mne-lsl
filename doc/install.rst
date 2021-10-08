.. _install:

====================
Install instructions
====================

BSL requires Python version `3.6` or higher. BSL is available on
`GitHub <https://github.com/bsl-tools/bsl>`_ and on
`Pypi <https://pypi.org/project/bsl/>`_.

- **GitHub**: Clone the `main repository <https://github.com/bsl-tools/bsl>`_
  and install with ``setup.py``

  .. code-block:: console

      $ python setup.py install

  For the developer mode, use ``develop`` instead of ``install``.

- **Pypi**: Install BSL using ``pip``

  .. code-block:: console

      $ pip install bsl

- **Conda**: Not yet available.

=====================
Optional dependencies
=====================

BSL installs the following dependencies:

- `numpy <https://numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `mne <https://mne.tools/stable/index.html>`_
- `pylsl <https://github.com/labstreaminglayer/liblsl-Python>`_
- `pyqt5 <https://www.riverbankcomputing.com/software/pyqt/>`_
- `pyqtgraph <https://www.pyqtgraph.org/>`_

Additionnal functionalities requires:

- `pyserial <https://github.com/pyserial/pyserial>`_: for the trigger using an
  :ref:`arduino2lpt`.
- `vispy <https://vispy.org/>`_: for the `~bsl.StreamViewer` vispy backend.
