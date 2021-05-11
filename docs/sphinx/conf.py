# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme
from pathlib import Path
from sphinx.domains.python import PythonDomain
from skmultiflow._version import __version__ as neurodecode_version

sys.path.insert(0, Path(os.environ['NEUROD_ROOT']) /  'neurodecode' / 'stream_viewer')
sys.path.insert(0, Path(os.environ['NEUROD_ROOT']) /  'neurodecode' / 'stream_receiver')
sys.path.insert(0, Path(os.environ['NEUROD_ROOT']) /  'neurodecode' / 'stream_player')
sys.path.insert(0, Path(os.environ['NEUROD_ROOT']) /  'neurodecode' / 'stream_recorder')
sys.path.insert(0, Path(os.environ['NEUROD_ROOT']) /  'neurodecode' / 'triggers')
#sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'NeuroDecode'
copyright = '2020, Kyuhwa Lee, Arnaud Desvachez'
author = 'K. Lee, A. Desvachez'
# The short X.Y version.
version = neurodecode_version
# The full version, including alpha/beta/rc tags.
release = neurodecode_version

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [  'sphinx.ext.autodoc',
                'sphinx.ext.autosummary',
                'sphinx.ext.napoleon']

# Napoleon settings
napoleon_include_init_with_doc = False

# autodoc options
autodoc_default_options = {'members': True, 'inherited-members': True}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates','_build']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
#source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# autosummary
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {'collapse_navigation': False,
                      'navigation_depth': 3}

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['neurodecode.']

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/images/neurodecode.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'neurodecodedoc'

# Avoid misleading warning triggered by more than one target for cross-reference
class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if 'refspecific' in node:
            del node['refspecific']
        return super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode)


def setup(sphinx):
    sphinx.add_domain(PatchedPythonDomain, override=True)
