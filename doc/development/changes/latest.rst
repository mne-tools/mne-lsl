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

Version 1.10
============

- Add support for callbacks in :class:`~mne_lsl.stream.StreamLSL` objects with :meth:`~mne_lsl.stream.StreamLSL.add_callback` (:pr:`419` by `Mathieu Scheltienne`_)
- Add support for hashing of :class:`~mne_lsl.stream.StreamLSL` objects (:pr:`424` by `Mathieu Scheltienne`_)
- Build ``liblsl`` with the MSVC ``v142`` toolchain on Windows (:pr:`440` by `Mathieu Scheltienne`_)
