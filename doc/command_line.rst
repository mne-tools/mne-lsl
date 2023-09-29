.. include:: ./links.inc

Command line
============

A ``Player``, or the legacy :class:`~bsl.stream_viewer.StreamViewer`, can be
called from the command line. For each command, the flag ``-h`` or ``--help`` provides
additional information.

Player
------

An `MNE <mne stable_>`_ readable file can be streamed with a `~bsl.player.PlayerLSL`
with the command:

.. code-block:: console

    $ bsl_player file lsl

With the arguments:

* ``file`` (mandatory): :term:`file-like <python:file object>`, file to stream.
* ``-n``, ``--name`` (optional, default ``BSL-Player``): :class:`str`, name of the LSL
  stream.
* ``-c``, ``--chunk_size`` (optional, default ``16``): :class:`int`, number of samples
  pushed at once.

StreamViewer
------------

A legacy :class:`~bsl.stream_viewer.StreamViewer` can be opened with the command:

.. code-block:: console

    $ bsl_stream_viewer

With the arguments:

- ``-s``, ``--stream_name`` (optional): :class:`str`, name of the stream to connect to.

.. note::

    If ``stream_name`` is not provided, a prompt is displayed to select a stream among
    the available ones.

The :class:`~bsl.stream_viewer.StreamViewer` opens 2 windows:

- A controller to select the channels to plot and set different plotting parameters.
- A plotting window using the ``pyqtgraph`` backend displaying the signal in real-time.

.. image:: _static/stream_viewer/stream_viewer.gif
   :alt: StreamViewer
   :align: center
