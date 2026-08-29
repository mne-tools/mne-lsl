.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below, so the changelog page will have a link to
   the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: You can use the :pr:`xx` and :issue:`xx` role to x-ref to a GitHub PR
   or issue from this project.

:hide-toc:

.. include:: ./authors.inc

.. _latest:

Version 1.15
============

- Fix duplicate epochs acquired by :class:`~mne_lsl.stream.EpochsStream` when the mapping of an event onto the data stream timestamps changed between acquisitions (:pr:`565` by `Mathieu Scheltienne`_)
- Fix :class:`~mne_lsl.player.PlayerLSL` pushing an oversized back-dated chunk when looping on a file whose size is an exact multiple of ``chunk_size`` (:pr:`565` by `Mathieu Scheltienne`_)
- Restore macOS intel wheels (:pr:`565` by `Eric Larson`_)
- Add a ``recover`` argument to :meth:`~mne_lsl.stream.StreamLSL.connect` to disable silent recovery of lost streams (:pr:`565` by `Eric Larson`_)
- Fix an intermittent abort on inlet destruction by not closing the stream before destroying it, which could engage the ``liblsl`` stream recovery machinery whose cancellation races with the destruction (:pr:`565` by `Eric Larson`_)
- Remove the legacy ``StreamViewer``, replaced by a new Qt 6 viewer (:pr:`573` by `Mathieu Scheltienne`_)
- Remove ``pyqtgraph`` and ``qtpy`` from the core dependencies, ``import mne_lsl`` no longer imports Qt; the viewer dependencies are now gathered in the mutually-exclusive optional dependency groups ``mne-lsl[pyqt6]`` and ``mne-lsl[pyside6]`` (:pr:`573` by `Mathieu Scheltienne`_)
- Remove the ``-s``/``--stream`` argument of the ``mne-lsl viewer`` command, the new viewer discovers streams from within its graphical interface (:pr:`573` by `Mathieu Scheltienne`_)
- Implement :meth:`~mne_lsl.stream.StreamLSL.plot`, which opens the new viewer on a connected stream and blocks until the viewer is closed; the stream is borrowed and is left connected (:pr:`573` by `Mathieu Scheltienne`_)
- Fix :meth:`~mne_lsl.stream.StreamLSL.get_channel_units` omitting bad channels, which made the returned list shorter than the number of channels and thus impossible to index by channel; bad channels are now included, as the documented behavior of the ``picks`` argument already promised, matching :meth:`~mne_lsl.stream.StreamLSL.get_channel_types` and :meth:`~mne_lsl.player.PlayerLSL.get_channel_units` (:pr:`573` by `Mathieu Scheltienne`_)
- Fix :attr:`~mne_lsl.stream.StreamLSL.connected` raising :class:`AssertionError` when read while the acquisition thread is resetting the stream, e.g. after a lost stream; a partially reset stream now reads as not connected (:pr:`573` by `Mathieu Scheltienne`_)
- :meth:`~mne_lsl.stream.StreamLSL.disconnect` is now idempotent and returns without raising when the stream is already disconnected, e.g. after the acquisition thread reset it following a lost stream (:pr:`573` by `Mathieu Scheltienne`_)
- Add :attr:`~mne_lsl.stream.StreamLSL.disconnect_reason`, the exception which disconnected a stream, e.g. a ``LostError`` for a stream whose source went away; ``None`` while connected or after a clean disconnection (:pr:`573` by `Mathieu Scheltienne`_)
- Detect a lost stream in the viewer: the document freezes its viewport on the last frame, shows a notice, reconnects in the background and resumes automatically when the same stream returns with the same channels, or explains why it refused to; a *borrowed* stream, i.e. one opened with :meth:`~mne_lsl.stream.StreamLSL.plot`, is reconnected only when asked to, since a reconnection would drop the filters, callbacks and acquisition delay its owner set on it (:pr:`573` by `Mathieu Scheltienne`_)
