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
- Improve LSL trigger by reducing the buffer, optimizing the execution, and limiting the data type to ``np.int8`` (:pr:`72`)
- Improve detection of Arduino to LPT converter and of parallel port (:pr:`72`)
- Improve error checking for integers and paths (:pr:`73`)
- Add a detrending option and additional y-ranges to the `~bsl.StreamViewer` (:pr:`88`)

Bugs
----

- Change type of the data stream from a `~bsl.StreamPlayer` to ``np.float64`` (:pr:`71`)
- Improve logger integration with sphinx-gallery (:pr:`87`)
- Set pins to 0 during the initialization of a parallel port (:pr:`86`)

API and behavior changes
------------------------

- Add the low-level `bsl.lsl` module re-implementing ``pylsl`` main objects `~bsl.lsl.StreamInfo`, `~bsl.lsl.StreamInlet`, `~bsl.lsl.StreamOutlet` (:pr:`71`)
- Remove ``bsl.triggers.SoftwareTrigger`` that are both imprecise and not needed with the future recorder (:pr:`72`)
- Remove the ``externals`` module and move the parallel port I/O code in the ``triggers`` module (:pr:`72`)

Authors
-------

* `Mathieu Scheltienne`_
