# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

import bsl

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.append(Path(__file__).parent.parent / 'bsl')

# -- Project information -----------------------------------------------------

project = 'BSL'
td = datetime.now(tz=timezone.utc)
copyright = f'{td.year}, Kyuhwa Lee, Arnaud Desvachez, Mathieu Scheltienne, '+\
    f'Last updated on {td.isoformat()}'
author = 'K. Lee, A. Desvachez, M. Scheltienne'

# The full version, including alpha/beta/rc tags
release = bsl.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
    'numpydoc'
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = '.rst'

# The main toctree document.
master_doc = 'index'

# List of documents that shouldn't be included in the build.
unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "py:obj"

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['bsl.']

# Generate autosummary even if no references
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'icon_links': [
        dict(name='GitHub',
             url='https://github.com/bsl-tools/bsl',
             icon='fab fa-github-square')
        ],
    'icon_links_label': 'Quick Links',  # for screen reader
    'use_edit_page_button': False,
    'navigation_with_keys': False,
    'show_toc_level': 1
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/icon-with-acronym/icon-with-acronym.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/icon/bsl-icon.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# -- Sphinx-gallery configuration --------------------------------------------

sphinx_gallery_conf = {
    'doc_module': 'bsl',
    'reference_url': dict(bsl=None),
    'examples_dirs': '../examples',
    'gallery_dirs': 'generated_examples',
    'plot_gallery': 'True',  # Avoid annoying Unicode/bool default warning
    'remove_config_comments': True,
    'abort_on_example_error': False,
    'filename_pattern': re.escape(os.sep),
    'line_numbers': False,
    'download_all_examples': False,
    'matplotlib_animations': True
    }

# -- Other extension configuration -------------------------------------------

# autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}

# Add intersphinx mappings
intersphinx_mapping = {
    'python': ('http://docs.python.org/3', None),
    'mne': ('https://mne.tools/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'vispy': ('https://vispy.org/', None),
    }

# numpydoc
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    'file-like': ':term:`file-like <python:file object>`',
    'path-like': ':term:`path-like`',
    'Path': ':class:`python:pathlib.Path`',
    'Process': ':class:`python:multiprocessing.Process`',
    # MNE
    'Info': 'mne.io.Info',
    'Raw': 'mne.io.Raw',
    # BSL
    'StreamReceiver': 'bsl.StreamReceiver',
    'StreamRecorder': 'bsl.StreamRecorder',
    'SoftwareTrigger': 'bsl.triggers.SoftwareTrigger'
    }

numpydoc_xref_ignore = {
    # words
    'instance', 'instances', 'of'
    }
