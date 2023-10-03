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

Version 1.0
===========

- Add :class:`mne_lsl.stream.StreamLSL` to connect to a numerical LSL Stream, automatically update an internal ringbuffer and provide an MNE-like Stream API (:pr:`93`)
- Add :class:`mne_lsl.player.PlayerLSL` to create a mock LSL stream from an MNE-readable file (:pr:`93`)
- Improve low-level LSL API :class:`mne_lsl.lsl.StreamInfo`, :class:`mne_lsl.lsl.StreamInlet`, :class:`mne_lsl.lsl.StreamOutlet` (:pr:`93`) compared to ``BSL`` 0.6.4
- Remove legacy and deprecated objects from ``BSL`` (:pr:`96`, :pr:`97`, :pr:`98`, :pr:`100`, :pr:`101`, :pr:`102`)

Authors
-------

* `Mathieu Scheltienne`_
