# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AnalysisG'
copyright = '2023, woywoy123'
author = 'woywoy123'
release = '0.2'
master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.imgmath",
    "breathe",
]

# Exclude the old deprecated documentation that's still in source/
exclude_patterns = [
    'core-classes',
    'core-structs',
    'getting-started',
    'gnn-training',
    'mc16-events',
    'pyc',
    'studies',
]

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {
    "AnalysisG": "../build/doxygen/xml"
}
breathe_default_project = "AnalysisG"
breathe_default_members = ('members', 'undoc-members')

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme = 'insegel'
html_theme = 'sphinx_rtd_theme'

# -- Run Doxygen before building Sphinx docs ---------------------------------
def run_doxygen(app):
    """Run doxygen before building sphinx documentation"""
    docs_dir = os.path.dirname(os.path.dirname(__file__))
    doxyfile = os.path.join(docs_dir, 'Doxyfile')
    if os.path.exists(doxyfile):
        subprocess.run(['doxygen', doxyfile], cwd=docs_dir, check=True)

def setup(app):
    app.connect('builder-inited', run_doxygen)

