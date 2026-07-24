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
