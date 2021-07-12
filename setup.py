#!/usr/bin/env python

import os

from pathlib import Path
from setuptools import setup

# Version
version = None
with open(Path(__file__).parent/'neurodecode'/'_version.py', 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

# Descriptions
short_description = """NeuroDecode real-time framework for online neuroscience research through LSL-compatible devices."""
with open('README.md', 'r') as file:
    long_description = file.read()


# Variables
NAME = 'NeuroDecode'
DESCRIPTION = short_description
LONG_DESCRIPTION = long_description
AUTHOR = 'Kyuhwa Lee, Arnaud Desvachez, Mathieu Scheltienne'
AUTHOR_EMAIL = 'lee.kyuh@gmail.com, arnaud.desvachez@gmail.com, mathieu.scheltienne@gmail.com'
URL = 'https://github.com/mscheltienne/NeuroDecode'
LICENSE = 'LGPL-2.1 - The GNU General Public License'
DOWNLOAD_URL = 'https://github.com/mscheltienne/NeuroDecode'
VERSION = version


# Dependencies
hard_dependencies = ('mne', 'numpy', 'pylsl', 'pyqt5', 'pyqtgraph', 'scipy')
install_requires = list()
with open('requirements.txt', 'r') as fid:
    for line in fid:
        req = line.strip()
        for hard_dep in hard_dependencies:
            if req.startswith(hard_dep):
                install_requires.append(req)

# Submodules
def package_tree(pkgroot):
    """Get the submodule list. Adapted from Vispy."""
    path = os.path.dirname(__file__)
    subdirs = [
        os.path.relpath(i[0], path).replace(os.path.sep, '.')
        for i in os.walk(os.path.join(path, pkgroot)) if '__init__.py' in i[2]]
    return sorted(subdirs)


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English'
        ],
    keywords='neuroscience neuroimaging EEG LSL real-time brain',
    project_urls={
        'Documentation': 'https://github.com/mscheltienne/NeuroDecode',
        'Source': 'https://github.com/mscheltienne/NeuroDecode'
        },
    platforms='any',
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=package_tree('neurodecode')
)
