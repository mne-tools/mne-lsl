==========
User-Guide
==========

In this section, we cover the core concepts in ``NeuroDecode``.

--------------
StreamReceiver
--------------
The base module for acquiring signals used by other modules such as StreamViewer and StreamRecorder. Go to API for more info.

------------
StreamViewer
------------
Visualize signals in real time with spectral filtering, common average filtering options and real-time FFT.

From a terminal:

.. code-block:: bash

    nd_stream_viewer

The following inputs can also be provided while launching:

- amp_name : The name of the LSL stream to visualize (Default is None)

.. code-block:: bash

    nd_stream_viewer amp_name

-------------
StreamRecoder
-------------
Record signals into fif format, a standard format mainly used in `MNE EEG analysis library <https://mne.tools/stable/index.html>`_.
The StreamRecorder can record in .pkl any stream types but it will convert to .fif only EEG streams.

From a terminal:

.. code-block:: bash

    nd_stream_recorder

The following inputs can also be provided while launching:

- record_dir : The path where to save the data
- amp_name : The LSL stream to record (Default: None, it will record all availbale streams)

.. code-block:: bash

    nd_stream_recorder record_dir amp_name

-------------
StreamPlayer
-------------
Replay recorded signals in real time as if it was transmitted from a real acquisition server.
The StreamPlayer can only replay data fif files.

From a terminal:

.. code-block:: bash

    nd_stream_player

The following inputs can also be provided while launching:

- amp_name : The name of the LSL stream
- fif_file : The path to the fif file
- chunk_size : The samples number sent in each chunk (Default is 16)
- trigger_file : The .ini file containing the mapping from int to string events (Default is None)

.. code-block:: bash

    nd_stream_player amp_name fif_file chunk_size trigger_file


-------
Decoder
-------
Contains decoder module. Currently, LDA, regularized LDA, Random Forests, and Gradient Boosting Machines are supported as the classifier type. Neural Network-based decoders are currently under experiment. The training of the decoder
needs to be implemented in the training protocol, you can base you work on the trainer_mi.py of the Motor Imagery protocol. Go to API for more info.

---------
Protocols
---------
Contains the  offline, training and online protocols for Motor Imagery (Brain-Computer Interface). New protocols should be added here in order to the GUI to detect them. The corresponding config files should also be added in the config_files folder. Google Glass visual feedback is supported through USB communication.

--------
Triggers
--------
Triggers are used to mark event (stimulus) timings during the recording. This module allows to send triggers through native LPT, Commercial USB2LPT adapter and Arduino convertor. It also supports Software triggers (saved in a txt file and added to the fif file at the end of the recording) and Fake triggers (no trigger really sent). Go to API for more info.


-----
Utils
-----
Contains various utilities. Go to API for more info.


