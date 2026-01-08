# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import subprocess

# -- Path setup --------------------------------------------------------------
# Add src directory to path for Python module documentation
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'AnalysisG'
copyright = '2024, AnalysisG Team'
author = 'AnalysisG Team'
version = '1.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'breathe',                    # Bridge between Doxygen and Sphinx
    'exhale',                     # Automatic API documentation from Doxygen XML
    'sphinx.ext.autodoc',         # Python autodoc
    'sphinx.ext.napoleon',        # Google/NumPy style docstrings
    'sphinx.ext.viewcode',        # Add links to source code
    'sphinx.ext.intersphinx',     # Link to other Sphinx docs
    'sphinx.ext.todo',            # Support TODO items
    'sphinx.ext.graphviz',        # Graphviz diagrams
    'recommonmark',               # Markdown support
]

# Templates path
templates_path = ['source/_templates']

# Patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'doxygen/html', 'doxygen/latex']

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['source/_static']
html_logo = None
html_favicon = None

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# -- Breathe configuration ---------------------------------------------------
# Path to Doxygen XML output
breathe_projects = {
    "AnalysisG": os.path.abspath("doxygen/xml")
}
breathe_default_project = "AnalysisG"
breathe_default_members = ('members', 'undoc-members')

# -- Exhale configuration ----------------------------------------------------
exhale_args = {
    # Root directory for generated RST files
    "containmentFolder":    "./source/api",
    "rootFileName":         "library_root.rst",
    "doxygenStripFromPath": "../src/AnalysisG",
    # Title for the root page
    "rootFileTitle":        "C++ API Reference",
    # File pattern matching
    "createTreeView":       True,
    # Run Doxygen before exhale
    "exhaleExecutesDoxygen": False,  # We'll run Doxygen separately
    # Full toctree listing
    "fullToctreeMaxDepth":  2,
    # Customizations
    "afterTitleDescription": """
This section contains the auto-generated API documentation for the AnalysisG framework.
The documentation is organized by classes, files, and namespaces.

**Main Entry Points:**

- :ref:`analysis <exhale_class_classanalysis>` - Central analysis orchestrator
- Event Templates - Base class for event data
- Graph Templates - Base class for graph construction
- Model Templates - Base class for ML models
""",
    # Collapsible trees
    "listingExclude": [r".*detail.*"],
}

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- TODO configuration ------------------------------------------------------
todo_include_todos = True

# -- GraphViz configuration --------------------------------------------------
graphviz_output_format = 'svg'

# -- Additional configuration ------------------------------------------------
# Suppress warnings about missing references that we know don't exist
nitpicky = False

# Add any custom CSS/JS
def setup(app):
    app.add_css_file('custom.css')
