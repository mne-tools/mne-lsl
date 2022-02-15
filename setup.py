#!/usr/bin/env python

import itertools

from pathlib import Path
from setuptools import setup, find_packages


# Version
version = None
with open(Path(__file__).parent/'bsl'/'_version.py', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break

if version is None:
    raise RuntimeError('Could not determine version.')


# Descriptions
short_description = """BrainStreamingLayer real-time framework for online neuroscience research through LSL-compatible devices."""
long_description_file = Path(__file__).parent / 'README.md'
with open(long_description_file, 'r') as file:
    long_description = file.read()
if long_description_file.suffix == '.md':
    long_description_content_type='text/markdown'
elif long_description_file.suffix == '.rst':
    long_description_content_type='text/x-rst'
else:
    long_description_content_type='text/plain'


# Variables
NAME = 'bsl'
DESCRIPTION = short_description
LONG_DESCRIPTION = long_description
LONG_DESCRIPTION_CONTENT_TYPE=long_description_content_type
AUTHOR = 'Kyuhwa Lee, Arnaud Desvachez, Mathieu Scheltienne'
AUTHOR_EMAIL = 'lee.kyuh@gmail.com, arnaud.desvachez@gmail.com, mathieu.scheltienne@gmail.com'
MAINTAINER = 'Mathieu Scheltienne'
MAINTAINER_EMAIL = 'mathieu.scheltienne@gmail.com'
URL = 'https://github.com/bsl-tools/bsl'
LICENSE = 'LGPL-2.1 - The GNU General Public License'
DOWNLOAD_URL = 'https://github.com/bsl-tools/bsl'
VERSION = version


# Dependencies
def get_requirements(path):
    """Get mandatory dependencies from file."""
    install_requires = list()
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            req = line.strip()
            if len(line) == 0:
                continue
            install_requires.append(req)

    return install_requires


def get_requirements_extras(path, add_all=True):
    """Map extra dependencies to the functionalities."""
    extras_require = dict()
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            req, tags = line.split('#')
            for tag in tags.split(','):
                tag = tag.strip()
                if tag not in extras_require:
                    extras_require[tag] = [req.strip()]
                else:
                    extras_require[tag].append(req.strip())

    # add tag 'all' at the end
    if add_all:
        extras_require['all'] = set(itertools.chain(*extras_require.values()))

    return extras_require


install_requires = get_requirements('requirements.txt')
extras_require = get_requirements_extras(
    'requirements_extras.txt', add_all=True)


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English'
        ],
    keywords='neuroscience neuroimaging EEG LSL real-time brain',
    project_urls={
        'Documentation': 'https://bsl-tools.github.io/',
        'Source': 'https://github.com/bsl-tools/bsl',
        'Tracker': 'https://github.com/bsl-tools/bsl/issues'
        },
    platforms='any',
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'bsl = bsl.commands.main:run',
            'bsl_stream_player = bsl.commands.bsl_stream_player:run',
            'bsl_stream_recorder = bsl.commands.bsl_stream_recorder:run',
            'bsl_stream_viewer = bsl.commands.bsl_stream_viewer:run'
          ]
        }
    )
