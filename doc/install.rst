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

=====================
Installation Example
=====================

Python version `3.8.10`

First of all, make sure that the path to the desired version of Python is present in the `Path` Environment Variable:

- Right click on `This PC` App and select `Properties`
- Scroll down the `About` section and select `Advanced system settings`
.. image:: _static\install\Advanced_system_settings.png
   :width: 600
- Click on `Environmental Variables` and verify that the the Python path (ex. C:\Users\User\Programs\Python\Python38\Scripts) is in the Environmental system variable `Path`.
.. image:: _static\install\Environmental_variables.png
   :width: 600
From the Terminal go to the working directory and clone the `main repository <https://github.com/bsl-tools/bsl>`_:

    .. code-block:: console

        $ git clone https://github.com/bsl-tools/bsl

Enter to `bsl` folder:

    .. code-block:: console
    
       $ cd bsl
       
Install bsl with ``setup.py`` in developer mode:

    .. code-block:: console
    
       $ python setup.py develop

Check that everything works:

- Download a dataset:

    .. code-block:: console
    
       $ import bsl
       $ bsl.datasets.eeg_resting_state.data_path()

- Run bsl stream player:
    
    .. code-block:: console
    
       $ cd C:\Users\User
       $ cd bsl_data
       $ cd eeg_sample
       $ bsl_stream_player testStream resting_state-raw.fif

- Run bsl stream viewer in a different terminal:

    .. code-block:: console
    
       $ bsl_stream _viewer


The stream viewer should be visualize:

.. image:: _static\install\bsl_stream_viewer.png
   :width: 600
   
    
    