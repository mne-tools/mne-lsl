.. include:: ./links.inc

API References
==============

Most-used classes
-----------------

.. currentmodule:: bsl

The main objects offer efficient communication with numerical LSL streams. A
`~bsl.Stream` uses an `MNE <mne stable_>`_-like API to efficiently interacts with a
numerical LSL stream. A `~bsl.Player` can mock an LSL stream from any
`MNE <mne stable_>`_ readable file.

.. autosummary::
   :toctree: ./generated/
   :nosignatures:

    Stream
    Player

LSL (low-level)
---------------

.. currentmodule:: bsl.lsl

If the high-level API is not sufficient, the low-level API interface directly with
`liblsl <lsl lib c++_>`_, with a similar API to `pylsl <lsl python_>`_.

Compared to `pylsl <lsl python_>`_, ``bsl.lsl`` pulls a chunk of *numerical* data faster
thanks to ``numpy``. In numbers, pulling a 1024 samples chunk with 65 channels in double
precision (``float64``) to python takes:

* 7.87 ms ± 58 µs with ``pylsl``
* 4.35 µs ± 134 ns with ``bsl.lsl``

More importantly, ``pylsl`` pulls a chunk in linear time ``O(n)``, scalings with the
number of samples; while ``bsl.lsl`` pulls a chunk in constant time ``O(1)``.

.. autosummary::
   :toctree: ./generated/
   :nosignatures:

   StreamInfo
   StreamInlet
   StreamOutlet
   library_version
   protocol_version
   local_clock
   resolve_streams

Triggers
--------

.. currentmodule:: bsl.triggers

Triggers are commonly used in combination with LSL to mark events in time.

.. autosummary::
   :toctree: ./generated/
   :nosignatures:

   MockTrigger
   LSLTrigger
   ParallelPortTrigger

Utilities
---------

.. currentmodule:: bsl

Logging utilities are available to interact with ``bsl`` logger.

.. autosummary::
    :toctree: ./generated/
    :nosignatures:

    add_file_handler
    set_log_level

Development utilities are available to help debug a setup.

.. autosummary::
    :toctree: ./generated/
    :nosignatures:

    sys_info

Legacy
------

.. currentmodule:: bsl.stream_viewer

Legacy classes and functions will be replaced with backward incompatible equivalent in
future versions.

.. autosummary::
   :toctree: ./generated/
   :nosignatures:

   StreamViewer
