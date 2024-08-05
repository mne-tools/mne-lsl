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

Version 1.5
===========

- Add ``exclude='bads'`` argument to :meth:`~mne_lsl.stream.StreamLSL.get_data` to control exclusion from ``picks`` (:pr:`258` by `Mathieu Scheltienne`_)
- Add :class:`~mne_lsl.stream.EpochsStream` to epoch a Stream object object on the fly (:pr:`258` by `Mathieu Scheltienne`_)

Infrastructure
--------------

- Improve unit tests by setting a marker ``slow`` on tests which take more than 5 seconds to complete (:pr:`258` by `Mathieu Scheltienne`_)
- Improve documentation build and unit test by adding a github action to retry a step when needed (:pr:`258` by `Mathieu Scheltienne`_)
