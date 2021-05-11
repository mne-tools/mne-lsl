============
Installation
============

Here we list the multiple options available to install ``NeuroDecode``.

**Notice:** ``NeuroDecode`` has been tested with **Python 3.7** only. 
We advise you to install Python thanks to `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `conda <https://docs.conda.io/en/latest/>`_. 

------------
Dependencies
------------

``NeuroDecode`` requires the use of external libraries:

* `numpy <www.numpy.org>`_
* `scipy <https://www.scipy.org/>`_
* `scikit learn <https://scikit-learn.org/stable/>`_
* `mne <https://mne.tools/stable/index.html>`_
* `pylsl <https://pypi.org/project/pylsl/>`_
* ...

The full list of dependencies can be found in setup.py.

------------
Requirements
------------

During the installation, ``NeuroDecode`` requires to provide two folders paths for:

    * the protocols' scripts and the subjects' config files 
    * the subjects' data

Create those two folders before starting the installation.

---------------------
Option 1. From source
---------------------

**Notice:** This installation requires  `pip <https://pip.pypa.io/en/stable/>`_. It is included in the installation of `Python <https://www.python.org/>`_, `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `conda <https://docs.conda.io/en/latest/>`_. If not reconized, add its path to the environment path.

#. Copy or clone ``NeuroDecode`` from the `project's GitHub page <https://github.com/fcbg-hnp/NeuroDecode>`_

#. From the ``NeuroDecode`` folder, run:

    * For Windows:

    .. code-block:: batch

       $ .\INSTALLER.cmd

    * For Linux:

    .. code-block:: bash

       $ ./INSTALLER.sh

   You will be asked to provide the absolute paths of the two previously created folders.

---------------------------
Option 2. Install from PyPI
---------------------------

Not yet available


----------------------------------
Option 3. Install from conda-forge
----------------------------------

Not yet available

-----------------------------
Option 3. Install with Docker
-----------------------------

Not yet available
