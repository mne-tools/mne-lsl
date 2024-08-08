---
title: 'Pycrostates: a Python library to study EEG microstates'
tags:
  - Python
  - neuroscience
  - neuroimaging
  - real-time application
  - lab streaming layer
  - EEG
  - MEG
  - brain
authors:
  - name: Mathieu Scheltienne
    orcid: 0000-0001-8316-7436
    affiliation: 1
  - name: Eric Larson
    affiliation: 2
    orcid: 0000-0003-4782-5360
  - name: Arnaud Desvachez
    affiliation: 1
  - name: Kyuhwa Lee
    orcid: 0000-0002-3854-4690
    affiliation: 3
affiliations:
 - name: Human Neuroscience Platform, Fondation Campus Biotech Geneva, Geneva, Switzerland
   index: 1
 - name: Institute for Learning & Brain Sciences, University of Washington, Seattle, WA, USA
   index: 2
 - name: Wyss Center for Bio and Neuroengineering, Geneva, Switzerland
   index: 3
date: 08 August 2024
bibliography: paper.bib
---

# Summary

Neural, physiological, and behavioral data are crucial in understanding the complexities
of the human brain and body. This data encompasses a wide range of measurements
including electrical activity of the brain (EEG), heart rate, muscle activity,
eye movements, and various other physiological signals. These measurements provide
insights into cognitive processes, emotional states, physical health, and behavioral
responses.

Real-time applications of this data, such as brain-computer interfaces, neurofeedback,
and real-time health monitoring, present significant challenges. These applications
require the continuous collection, synchronization, and processing of data from multiple
sources with minimal latency. Additionally, real-time applications must address the
complexities of instantaneous data handling, including noise reduction and artifact
management.

The Lab Streaming Layer (LSL) [@labstreaminglayer] is an open-source framework designed
to facilitate the collection, synchronization, and transmission of time-series data from multiple devices in real-time. LSL provides a common protocol for streaming data,
allowing researchers to focus on data analysis without needing to manage the specific
APIs of various devices. By using LSL, data from different sources can be streamed, synchronized, and accessed through a unified interface. This streamlines the research process, enhances interoperability, and reduces the complexity associated with
integrating diverse data streams.

MNE-LSL significantly improves the integration of LSL with MNE-Python [@GramfortEtAl2013a], a comprehensive tool for processing and analyzing
electrophysiological data.

# Statement of need

In recent years, the demand for real-time acquisition and processing of neural, physiological, and behavioral data has surged, driven by the increasing popularity of brain-computer interfaces (BCIs) and neurofeedback applications. As BCIs and
neurofeedback become more accessible and sophisticated, researchers and developers are seeking robust, flexible tools to efficiently handle the complex, real-time data streams required for these applications.

OpenBCI [@openbci] and BrainFlow [@brainflow] are popular platforms designed primarily
for real-time data acquisition and processing. These tools are closely integrated with
the OpenBCI hardware ecosystem, offering a seamless experience for users working with OpenBCI boards. While this tight integration simplifies the setup for users of OpenBCI hardware, it also
limits the flexibility for researchers who wish to integrate and synchronize data from multiple hardware sources beyond the OpenBCI ecosystem.

Neuromore [@neuromore] is another platform that provides a user-friendly graphical interface for real-time data processing. It is designed to cater to users with minimal programming expertise, offering pre-built modules for data acquisition, processing, and
visualization. However, this GUI-based approach inherently restricts the customization
and flexibility that more advanced users might require.

In contrast to these more hardware-specific or restrictive platforms, the Lab Streaming Layer (LSL) has gained wide adoption in the EEG and broader neurophysiological research communities due to its device-agnostic framework. The widespread use of LSL in the EEG world stems from its ability to abstract away the complexities associated with different device APIs, providing a unified protocol for data streaming. Additionally, LSL streams are accessible in a variety of programming languages, including MATLAB, Python, C#,
Unity, and C++, among others. This cross-language compatibility further enhances its flexibility, enabling researchers to develop custom tools and applications in their preferred programming environment.

Accessing real-time LSL streams in Python is facilitated by several tools, each offering different levels of abstraction and ease of use. The most direct method is through
pylsl, a low-level API that provides fine-grained control over LSL streams. While pylsl
is powerful, its low-level nature can make it challenging for non-developers to work
with.

To address the need for a more user-friendly interface, mne-realtime was developed as an extension of MNE-Python. Originally, mne-realtime was designed to interface with the FieldTrip buffer and the RtClient for Neuromag systems, allowing for real-time data processing within the MNE-Python framework. Although support for LSL streams was eventually incorporated, the integration was not as intuitive as it could have been, leading to limited adoption within the broader MNE-Python community.

With the increasing popularity and wide adoption of LSL in the neurophysiological community, the maintenance and development focus shifted away from mne-realtime in favor of mne-lsl. MNE-LSL integrates more seamlessly with MNE-Python, offering a more
intuitive and accessible interface for researchers working with real-time LSL streams. Additionally, mne-lsl re-implements the low-level pylsl library more efficiently using NumPy [@harris2020array] and automatically fetches the correct liblsl library for the platform it is run on. These enhancements ensure that mne-lsl provides a robust and user-friendly solution for real-time data processing within the MNE-Python ecosystem.

# Acknowledgements



# References
