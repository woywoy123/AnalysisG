import os
import sys
import subprocess
from pathlib import Path

# Path setup
sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'AnalysisG'
copyright = '2023, AnalysisG Team'
author = 'AnalysisG Team'
version = '5.0'
release = '5.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'autodocsumm',
    'breathe',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
]

# Breathe configuration
breathe_projects = {
    "AnalysisG": "../../docs/doxygen/xml"
}
breathe_default_project = "AnalysisG"
breathe_domain_by_extension = {
    "h": "cpp",
    "cxx": "cpp",
    "cu": "cpp",
    "cuh": "cpp",
    "py": "py",
    "pyx": "py",
    "pxd": "py",
}

# Run doxygen if on Read the Docs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    subprocess.call('cd ../.. && doxygen', shell=True)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames
source_suffix = ['.rst', '.md']

# The main toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# Custom stylesheet
def setup(app):
    app.add_css_file('custom.css')

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'root': ('https://root.cern.ch/doc/master/', None),
}