# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AnalysisG'
copyright = '2023, woywoy123'
author = 'woywoy123'
release = '0.2'
master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.imgmath", "breathe"]

# Breathe Configuration for Doxygen integration
breathe_projects = {
    "AnalysisG": "../doxygen-docs/xml/"
}
breathe_default_project = "AnalysisG"
breathe_default_members = ('members', 'undoc-members')
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme = 'insegel'
html_theme = 'sphinx_rtd_theme'

