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

Version 0.6
===========

Enhancements
------------

- Improve GitHub workflows and add CodeCov (:pr:`71`)
- Add `bsl.sys_info` to display system information and dependency version  (:pr:`71`)

Bugs
----

- Change type of the data stream from a `bsl.StreamPlayer` to ``np.float64`` (:pr:`71`)

API and behavior changes
------------------------

- Add the low-level `bsl.lsl` module re-implementing ``pylsl`` main objects `~bsl.lsl.StreamInfo`, `~bsl.lsl.StreamInlet`, `~bsl.lsl.StreamOutlet` (:pr:`71`)

Authors
-------

* `Mathieu Scheltienne`_
