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

The Lab Streaming Layer (LSL) is an open-source framework designed to facilitate the
collection, synchronization, and transmission of time-series data from multiple devices
in real-time. LSL provides a common protocol for streaming data, allowing researchers to
focus on data analysis without needing to manage the specific APIs of various devices.
By using LSL, data from different sources can be streamed, synchronized, and accessed
through a unified interface. This streamlines the research process, enhances
interoperability, and reduces the complexity associated with integrating diverse data
streams.

MNE-LSL significantly improves the integration of LSL with MNE-Python [from v1.4.2; @mne-python_2013], a comprehensive tool for processing and analyzing
electrophysiological data.

# Statement of need



# Acknowledgements



# References
