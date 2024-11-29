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

Version 1.8
===========

- Fix epoching with :class:`~mne_lsl.stream.EpochsStream` which was erroneously resetting the property ``n_new_samples`` from the attached :class:`~mne_lsl.stream.StreamLSL` objects (:pr:`371` by `Mathieu Scheltienne`_)
