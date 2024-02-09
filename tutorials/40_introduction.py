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
Source: `LabStreamingLayer website <lsl_>``_.

In every real-time application, a server emits a data stream and one or multiple clients
connect to the server and receive the data. In the LSL terminology, the server is called
a :class:`~mne_lsl.lsl.StreamOutlet` and the client is called a
:class:`~mne_lsl.lsl.StreamInlet`. The strength of LSL lies in the interoperability and
synchronization of the streams. A client can connect to multiple servers, running on the
same or on different computers (and thus different platforms/OS), and synchronize the
streams from the different servers.

MNE-LSL further abstracts the LSL API to provide a high-level API similar to
`MNE <mne stable_>`_ API. This tutorial focuses on this high-level API, while the
low-level LSL api is described in #TODO x-ref tutorial low-level API.

Conceptually, a real-time LSL stream can be considered as a
continuous recording (:class:`mne.io.Raw`) with an unknown length and with only access
to the current and past samples.
"""
