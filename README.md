[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/mne-tools/mne-lsl/graph/badge.svg?token=Xoeh6T13qi)](https://codecov.io/gh/mne-tools/mne-lsl)
[![tests](https://github.com/mne-tools/mne-lsl/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/mne-tools/mne-lsl/actions/workflows/pytest.yml)
[![doc](https://github.com/mne-tools/mne-lsl/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/mne-tools/mne-lsl/actions/workflows/doc.yml)

<img align="right" src="https://raw.githubusercontent.com/mne-tools/mne-lsl/main/doc/_static/logos/logo-mne-hex.svg" alt="logo" width="200"/>

**MNE-LSL** [(Documentation website)](https://mne.tools/mne-lsl)
provides a real-time brain signal streaming framework.
**MNE-LSL** contains an improved python-binding for the Lab Streaming Layer C++ library,
`mne_lsl.lsl`, replacing `pylsl`. This low-level binding is used in high-level objects
to interact with LSL streams.

Any signal acquisition system supported by native LSL or OpenVibe is also
supported by MNE-LSL. Since the data communication is based on TCP, signals can be
transmitted wirelessly. For more information about LSL, please visit the
[LSL github](https://github.com/sccn/labstreaminglayer).

# Install

MNE-LSL supports `python ≥ 3.9` and is available on
[PyPI](https://pypi.org/project/mne-lsl/).
Install instruction can be found on the
[documentation website](https://mne.tools/mne-lsl/stable/install.html).

# Acknowledgment

<img align="right" src="https://raw.githubusercontent.com/mne-tools/mne-lsl/main/doc/_static/partners/fcbg-hnp-meeg.png" width=150>

**MNE-LSL** is based on **BSL** and **NeuroDecode**. The original version developed by
[**Kyuhwa Lee**](https://github.com/dbdq) was recognised at
[Microsoft Brain Signal Decoding competition](https://github.com/dbdq/microsoft_decoding)
with the First Prize Award (2016).
**MNE-LSL** is based on the refactor version, **BSL** by
[**Mathieu Scheltienne**](https://github.com/mscheltienne) and
[**Arnaud Desvachez**](https://github.com/dnastars) for the
[Fondation Campus Biotech Geneva (FCBG)](https://github.com/fcbg-hnp-meeg) and
development is still supported by the
[Human Neuroscience Platform (FCBG)](https://hnp.fcbg.ch/).

# Copyright and license

The code is released under the
[BSD 3-Clause License](https://opensource.org/license/bsd-3-clause/).
