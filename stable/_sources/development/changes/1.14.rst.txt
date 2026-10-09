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

Version 1.14
============

- Add alternative to stream annotations on a single string typed channel (:pr:`554` by `Alin G. Chitu`_)
- Add support for recording a :class:`~mne_lsl.stream.StreamLSL` to disk with :meth:`~mne_lsl.stream.StreamLSL.start_record` and :meth:`~mne_lsl.stream.StreamLSL.stop_record` (:pr:`562` by `Mathieu Scheltienne`_)
