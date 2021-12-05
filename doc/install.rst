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

Additional functionalities requires:

- `pyserial <https://github.com/pyserial/pyserial>`_: for the trigger using an
  :ref:`arduino2lpt`.
- `vispy <https://vispy.org/>`_: for the `~bsl.StreamViewer` vispy backend.

=============================
Installation and test example
=============================

| Python version ``3.8.10``
| Operating System: ``Windows 10``

**Step 1:** Make sure that the path to the Python installation folder
``Python38`` and ``Python38\Scripts`` present in the ``Path`` environment
variable of the PC:

- In the start menu, right click on ``This PC`` App and select ``Properties``
- Under ``Related settings``, click on ``Advanced system settings``

.. image:: _static/install/advanced_system_settings.png
   :align: center
   :width: 400

- Select ``Environmental Variables``

.. image:: _static/install/environmental_variables.png
   :align: center
   :width: 300

- Check that the the Python installation folder path (e.g.
  ``C:\Program Files\Python38\``) and the Python scripts folder (e.g.
  ``C:\Program Files\Python38\Scripts\``) are in the environment variable
  ``Path``, either user or system-wide.

| **Step 2:** Install BSL with ``pip``:

  .. code-block:: console

      $ pip install bsl

Alternatively clone the main repository in the current directory and install
BSL:

  .. code-block:: console

      $ git clone https://github.com/bsl-tools/bsl
      $ cd bsl
      $ python setup.py install

**Step 3:** Check that everything works:

- Test that ``pylsl`` was correctly installed with the core ``liblsl``.

  .. code-block:: python

      import pylsl

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
