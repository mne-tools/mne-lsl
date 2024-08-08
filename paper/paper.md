---
title: 'MNE-LSL: Real-time framework integrated with MNE-Python for online neuroscience research through LSL-compatible devices.'
tags:
  - Python
  - neuroscience
  - neuroimaging
  - real-time application
  - lab streaming layer
  - EEG
  - MEG
  - brain
  - electrophysiology
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
 - name: Fondation Campus Biotech Geneva, Geneva, Switzerland
   index: 1
 - name: Institute for Learning & Brain Sciences, University of Washington, Seattle, WA, USA
   index: 2
 - name: Wyss Center for Bio and Neuroengineering, Geneva, Switzerland
   index: 3
date: 08 August 2024
bibliography: paper.bib
---

# Summary

Neural, physiological, and behavioral data plays a key-role in understanding the human
brain and body. This data encompasses a wide range of measurements including electrical
activity of the brain (EEG), heart rate, muscle activity, eye movements. These
measurements provide insights into cognitive processes, emotional states, physical
health, and behavioral responses.

With the growing popularity of brain-computer interfaces (BCIs) and neurofeedback
applications, the demand for real-time processing of neurophysiological data has surged.
Real-time applications present unique challenges, requiring the continuous collection,
synchronization, and processing of data from multiple sources with minimal latency. This
requires the implementation of advanced methods for handling data in real-time,
including the reduction of noise and management of artifacts.

A critical challenge in real-time applications is the need for direct access to data
streams from measurement devices, which often rely on device-specific APIs. Adapting a
real-time application from one device to another can be labor-intensive, as it requires
modifications to the entire communication protocol. To address this issue, the Lab
Streaming Layer (LSL) [@labstreaminglayer] offers a standardized protocol for streaming
time-series data from multiple devices in real-time. LSL has gained significant
popularity, particularly among EEG manufacturers, many of whom now provide LSL streaming
capabilities out-of-the-box with their devices. By abstracting the complexities of
device-specific APIs, LSL allows researchers to concentrate on data analysis rather than
the intricacies of device communication.

Beyond data acquisition, the real-time analysis of signals also presents significant
challenges. MNE-Python [@GramfortEtAl2013a] is a comprehensive toolset for processing
and analyzing electrophysiological data in Python, a widely-used programming language in
the scientific community. MNE-LSL enhances the integration of LSL with MNE-Python,
providing robust objects for handling and processing both continuous and epoch-based
data from real-time sources, thereby streamlining the development of real-time
neurophysiological applications.

# Statement of need

The rise of brain-computer interfaces (BCIs) and neurofeedback applications has led to
an increased demand for tools capable of real-time acquisition and processing of neural,
physiological, and behavioral data. As these applications become more sophisticated and
widespread, researchers and developers require flexible, robust solutions to handle the
complexities of real-time data streams, which often involve multiple devices and
modalities.

Existing platforms like OpenBCI [@openbci] and BrainFlow [@brainflow] are tailored for
specific hardware ecosystems, offering streamlined workflows for users within those
systems. However, their tight integration with proprietary hardware can limit
flexibility and adaptability, particularly for researchers seeking to work with a
diverse array of measurement devices. Similarly, GUI-based platforms like Neuromore
[@neuromore] prioritize user-friendliness, often at the expense of customization and
advanced functionality, which can be a limitation for experienced users requiring more
control over data processing pipelines.

In contrast, the Lab Streaming Layer (LSL) has emerged as a widely adopted,
device-agnostic protocol within the neurophysiological research community, especially
for EEG data. LSL's ability to unify data streaming from various devices under a common
framework has made it an invaluable tool for researchers who need to integrate multiple
data sources seamlessly. However, while LSL provides a solid foundation for real-time
data acquisition, its integration with Python, particularly for processing and analysis,
has been less intuitive and less accessible.

To bridge this gap, MNE-LSL was developed as a solution that not only facilitates the
acquisition of real-time LSL streams but also integrates seamlessly with MNE-Python, a
leading toolset for electrophysiological data analysis. Unlike its predecessor,
MNE-realtime [@mne-realtime], which had limitations in LSL support and user interface
design, MNE-LSL offers an intuitive, MNE-Python-like API, allowing for real-time
processing of continuous and epoch-based data streams.

MNE-LSL further enhances the real-time data processing workflow by re-implementing the
low-level pylsl library with more efficient NumPy [@harris2020array] operations and
ensuring compatibility across platforms through automatic handling of the liblsl
library. These improvements provide researchers with a powerful, user-friendly tool for
real-time data analysis within the Python ecosystem, meeting the growing needs of the
neurophysiological research community.

# Acknowledgements

We would like to acknowledge the the LSL developers, with special thanks to Tristan
Stenner and Chadwick Boulay.

MNE-LSL development is supported by the Fondation Campus Biotech Geneva, Geneva,
Switzerland.

# References
