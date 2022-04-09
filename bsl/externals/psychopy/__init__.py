#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Part of the PsychoPy library
# Copyright (C) 2002-2018 Jonathan Peirce (C) 2019-2022 Open Science Tools Ltd.
# Distributed under the terms of the GNU General Public License (GPL).

import sys

__version__ = '2022.1.2'
__license__ = 'GNU GPLv3 (or more recent equivalent)'
__author__ = 'Open Science Tools Ltd'
__author_email__ = 'support@opensciencetools.org'
__maintainer_email__ = 'support@opensciencetools.org'
__url__ = 'https://www.psychopy.org/'
__download_url__ = 'https://github.com/psychopy/psychopy/releases/'
__git_sha__ = 'n/a'
__build_platform__ = 'n/a'

__all__ = []


if sys.version_info.major < 3:
    raise ImportError("psychopy does not support Python2 installations. "
                      "The last version to support Python2.7 was PsychoPy "
                      "2021.2.x")
