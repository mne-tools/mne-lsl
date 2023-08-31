.. _cli:

Command-Line
============

BSL propose to use 3 of its main classes from the command line:

- :ref:`stream_player`
- :ref:`stream_viewer`

For each command, the flag ``-h`` can be used to obtain additional information.

.. _stream_player:

Player
------

The :class:`~bsl.Player` can be called from the command-line with:

.. code-block:: console

    $ bsl_stream_player file -n stream_name

With the positional arguments:

- ``file``: :term:`file-like <python:file object>`

With the optional arguments:

- ``-n``, ``--name``: `str`, name of the LSL stream.
  Default ``BSL-Player``.

.. _stream_viewer:

StreamViewer
------------

The :class:`~bsl.stream_viewer.StreamViewer` can be called from the command-line with:

.. code-block:: console

    $ bsl_stream_viewer

With the optional arguments:

- ``-s``, ``--stream_name``: `str`, stream to visualize.

.. note::

    If ``stream_name`` is not provided, a prompt is displayed to select a
    stream among the available ones.

The :class:`~bsl.stream_viewer.StreamViewer` opens 2 windows:

- A controller to select the channels to plot and set different plotting
  parameters.
- A plotting window using the ``pyqtgraph`` backend displaying the signal in
  real-time.

.. image:: _static/stream_viewer/stream_viewer.gif
   :alt: StreamViewer
   :align: center
