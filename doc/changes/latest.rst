.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below, so the changelog page will have a link to
   the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: You can use the :pr:`xx` and :issue:`xx` role to x-ref to a GitHub PR
   or issue from this project.

.. include:: ./authors.inc

.. _latest:

Version 0.7
===========

Enhancements
------------

- xxx

Bugs
----

- Install linux dependencies during publication workflow
- Fix handling of processing flags in `~bsl.lsl.StreamInlet`

API and behavior changes
------------------------

- `bsl.lsl.StreamInlet.pull_sample` now returns an empty list or array if no sample is available instead of ``None``

Authors
-------

* `Mathieu Scheltienne`_
