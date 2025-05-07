# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import subprocess

# Path setup
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'AnalysisG'
copyright = '2023, AnalysisG Team'
author = 'AnalysisG Team'
release = '1.0.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'breathe',
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None  # Add your logo path here if you have one
html_favicon = None

# Breathe configuration for Doxygen integration
breathe_projects = {
    "AnalysisG": os.path.abspath("./doxygen/xml")
}
breathe_default_project = "AnalysisG"
breathe_domain_by_extension = {"py": "py"}

# Run Doxygen on Read the Docs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    subprocess.call('cd ..; doxygen docs/Doxyfile', shell=True)

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# Auto-generate API docs
autosummary_generate = True