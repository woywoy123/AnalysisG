# -- Project information -----------------------------------------------------
project = 'AnalysisG'
copyright = '2024, Dein Name'
author = 'Dein Name'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'breathe',
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
master_doc = 'index'

# -- Breathe Configuration --------------------------------------------------
breathe_projects = {
    "AnalysisG": "../doxygen-docs/xml"
}
breathe_default_project = "AnalysisG"

# -- Language ---------------------------------------------------------------
language = 'en'
