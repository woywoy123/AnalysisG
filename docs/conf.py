import os
import sys
import subprocess
from pathlib import Path

# Add the project source directory to the path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'AnalysisG'
copyright = '2023, AnalysisG Team'
author = 'AnalysisG Team'
release = '1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'breathe',
    'myst_parser',
]

# Breathe configuration
breathe_projects = {"AnalysisG": "../docs/xml"}
breathe_default_project = "AnalysisG"
breathe_domain_by_extension = {
    "h": "cpp",
    "cxx": "cpp",
    "cu": "cpp",
    "cuh": "cpp",
}

# Markdown configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Run doxygen on Read the Docs build
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    subprocess.call('cd ..; doxygen', shell=True)

# General configuration
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

# Include custom CSS
def setup(app):
    app.add_css_file('custom.css')