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

Version 1.7
===========

- Add the MNE configuration variable ``MNE_LSL_LIB_FOLDER`` to set the liblsl fetcher download directory (:pr:`350` by `Mathieu Scheltienne`_)
- Change default liblsl fetcher download directory to ``MNE_DATA / MNE-LSL-data / liblsl`` (:pr:`350` by `Mathieu Scheltienne`_)
