"""
Introduction to real-time LSL streams
=====================================

.. include:: ./../../links.inc

LSL is an open-source networked middleware ecosystem to stream, receive, synchronize,
and record neural, physiological, and behavioral data streams acquired from diverse
sensor hardware. It reduces complexity and barriers to entry for researchers, sensor
manufacturers, and users through a simple, interoperable, standardized API to connect
data consumers to data producers while abstracting obstacles such as platform
differences, stream discovery, synchronization and fault-tolerance.
Source: `LabStreamingLayer website <lsl_>`_.

In real-time applications, a server emits a data stream, and one or more clients connect
to the server to receive this data. In LSL terminology, the server is referred to as a
:class:~mne_lsl.lsl.StreamOutlet, while the client is referred to as a
:class:~mne_lsl.lsl.StreamInlet. The power of LSL resides in its ability to facilitate
interoperability and synchronization among streams. Clients have the capability to
connect to multiple servers, which may be running on the same or different computers
(and therefore different platforms/operating systems), and synchronize the streams
originating from these various servers.

MNE-LSL enhances the LSL API by offering a high-level interface akin to the
`MNE-Python <mne stable_>`_ API. While this tutorial concentrates on the high-level API,
detailed coverage of the low-level LSL API is provided in
`this separate tutorial <tut-low-level-api_>`_.

Concepts
--------

In essence, a real-time LSL stream can be envisioned as a perpetual recording, akin to
a :class:mne.io.Raw instance, characterized by an indeterminate length and providing
access solely to current and preceding samples. In memory, it can be depicted as a ring
buffer, also known as a circular buffer, a data structure employing a single, unchanging
buffer size, seemingly interconnected end-to-end.

.. image:: ../../_static/tutorials/circular-buffer-light.png
    :align: center
    :class: only-light

.. image:: ../../_static/tutorials/circular-buffer-dark.png
    :align: center
    :class: only-dark

Within a ring buffer, there are two pivotal pointers:

* The "head" pointer, also referred to as "start" or "read," indicates the subsequent
  data block available for reading.
* The "tail" pointer, known as "end" or "write," designates the forthcoming data block
  to be replaced with fresh data.

In a ring buffer configuration, when the "tail" pointer aligns with the "head" pointer,
data is overwritten before it can be accessed. Conversely, the "head" pointer cannot
surpass the "tail" pointer; it will always lag at least one sample behind. In all cases,
it falls upon the user to routinely inspect and fetch samples from the ring buffer,
thereby advancing the "head" pointer.

Within MNE-LSL, the :class:~mne_lsl.stream.StreamLSL object manages a ring buffer
internally, which is continuously refreshed with new samples. Notably, the two pointers
are concealed, with the head pointer being automatically adjusted to the latest received
sample. Given the preference for accessing the most recent information in neural,
physiological, and behavioral real-time applications, this operational approach
streamlines interaction with LSL streams and mitigates the risk of users accessing
outdated data.

Mocking an LSL stream
---------------------

To build real-time applications or showcase their functionalities, such as in this
tutorial, it's essential to generate simulated LSL streams. This involves creating a
:class:~mne_lsl.lsl.StreamOutlet and regularly sending data through it.
"""
