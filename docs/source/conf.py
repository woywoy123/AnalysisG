# Configuration file for the Sphinx documentation builder.
import os
import sys
import subprocess

# -- Project information -----------------------------------------------------
project = 'AnalysisG'
copyright = '2025, AnalysisG Team'
author = 'AnalysisG Team'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'breathe',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'
language = 'de'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {
    "AnalysisG": "../doxygen/xml/"
}
breathe_default_project = "AnalysisG"
breathe_default_members = ('members', 'undoc-members')

# Generiere Doxygen-Dokumentation beim Build
def generate_doxygen_xml(app):
    cwd = os.getcwd()
    try:
        # Navigate to project root
        os.chdir(os.path.join(cwd, "../.."))
        subprocess.call('doxygen Doxyfile', shell=True)
    finally:
        os.chdir(cwd)

def setup(app):
    app.connect("builder-inited", generate_doxygen_xml)
