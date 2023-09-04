.. include:: ../links.inc

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
number of samples; while ``bsl.lsl`` pulls a chunk in constant time ``O(1)``. Additional
details on the differences with ``pylsl`` and ``bsl`` can be found
:ref:`here<resources/pylsl:Differences with pylsl>`.

.. autosummary::
   :toctree: ../generated/api
   :nosignatures:

   StreamInfo
   StreamInlet
   StreamOutlet
   library_version
   protocol_version
   local_clock
   resolve_streams
