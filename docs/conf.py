# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full
# list see the documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'AnalysisG'
copyright = '2023'
author = 'AnalysisG Team'

# -- General configuration ---------------------------------------------------
extensions = [
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = []  # Removed '_static' to avoid warnings

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {"AnalysisG": "doxygen/xml"}
breathe_default_project = "AnalysisG"
breathe_default_members = ('members', 'undoc-members')

# -- Additional configuration ------------------------------------------------
master_doc = 'index'