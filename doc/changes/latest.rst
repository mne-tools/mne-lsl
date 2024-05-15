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

Version 1.4
===========

- Fix handling of warnings through the logger (:pr:`243` by `Mathieu Scheltienne`_)
- Fix handling of CLI arguments through :class:`~mne_lsl.player.PlayerLSL` entry-point (:pr:`246` by `Mathieu Scheltienne`_)
- Fix push operation by a :class:`~mne_lsl.player.PlayerLSL` with a ``chunk_size`` set to 1 to use :meth:`mne_lsl.lsl.StreamOutlet.push_sample` instead of :meth:`mne_lsl.lsl.StreamOutlet.push_chunk` (:pr:`257` by `Mathieu Scheltienne`_)
