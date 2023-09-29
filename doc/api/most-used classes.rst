.. include:: ../links.inc

Most-used classes
-----------------

The main objects offer efficient communication with numerical streams.

Stream
~~~~~~

A ``Stream`` uses an `MNE <mne stable_>`_-like API to efficiently interacts with a
numerical stream.

.. currentmodule:: bsl.stream

.. autosummary::
   :toctree: ../generated/api
   :nosignatures:

    StreamLSL

Player
~~~~~~

A ``Player`` can mock a real-time stream from any `MNE <mne stable_>`_ readable file.

.. currentmodule:: bsl.player

.. autosummary::
   :toctree: ../generated/api
   :nosignatures:

    PlayerLSL
