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
- Fix handling of CLI arguments through :class:`~mne_lsl.player.PlayerLSL` entry-point (:pr:`246`, :pr:`267` by `Mathieu Scheltienne`_)
- Change default download directory for the datasets from ``~/mne_data/MNE-LSL`` to ``~/mne_data/MNE-LSL-data`` to match other MNE datasets (:pr:`256` by `Mathieu Scheltienne`_)
- Add example for a simple real-time QRS R-peak detector (:pr:`256` by `Mathieu Scheltienne`_)
- Fix push operation by a :class:`~mne_lsl.player.PlayerLSL` with a ``chunk_size`` set to 1 to use :meth:`mne_lsl.lsl.StreamOutlet.push_sample` instead of :meth:`mne_lsl.lsl.StreamOutlet.push_chunk` (:pr:`257` by `Mathieu Scheltienne`_)
- Add example of :class:`~mne_lsl.player.PlayerLSL` run in a child process (:pr:`267` by `Mathieu Scheltienne`_)

API changes
-----------

- The :class:`~mne_lsl.player.PlayerLSL` default ``chunk_size`` is now set to 10 instead of 64 samples (:pr:`264` by `Mathieu Scheltienne`_)
- The ``Player`` and ``Stream`` objects now use a :class:`concurrent.futures.ThreadPoolExecutor` instead of single-use threads (:pr:`264` by `Mathieu Scheltienne`_)

Infrastructure
--------------

- Improve unit tests by (1) using a ``chunk_size`` of 200 samples in players, (2) running players in a separate process, (3) ensuring concurrent threads are not limited by one thread hogging the limited CI resources (:pr:`264` by `Mathieu Scheltienne`_)
