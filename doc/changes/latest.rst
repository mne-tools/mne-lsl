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

Version 1.2
===========

Enhancements
------------

- Apply MNE design philosophy by returning ``self`` in the methods modifying a :class:`~mne_lsl.stream.StreamLSL` and :class:`~mne_lsl.player.PlayerLSL` in-place (:pr:`200` by `Mathieu Scheltienne`_)
- Add argument ``annotations`` to :class:`~mne_lsl.player.PlayerLSL` to stream annotations on a separate :class:`~mne_lsl.lsl.StreamOutlet` (:pr:`202` by `Mathieu Scheltienne`_)
- Add support for a :class:`~mne.io.Raw` object as direct input to a :class:`~mne_lsl.player.PlayerLSL` (:pr:`202` by `Mathieu Scheltienne`_)

Bugs
----

- xxx

API and behavior changes
------------------------

- xxx

Authors
-------

* `Mathieu Scheltienne`_
