# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'AnalysisG'
copyright = '2023, AnalysisG Team'
author = 'AnalysisG Team'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'breathe',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {
    "AnalysisG": "./doxygen_output/xml"
}
breathe_default_project = "AnalysisG"