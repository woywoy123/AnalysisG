# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
project = "AnalysisG"
copyright = "2024, AnalysisG Contributors"
author = "AnalysisG Contributors"
release = "1.0"
version = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
}
html_show_sourcelink = False

# -- Breathe configuration ---------------------------------------------------
# Breathe reads the Doxygen XML output and bridges it into Sphinx.
# The Doxyfile writes XML to doxygen-docs/xml/ at the repo root.
# When Read the Docs runs this conf.py the pre_build step runs
# ``doxygen Doxyfile`` first, so the XML is always available.
# When conf.py is executed from docs/source/ we walk two levels up.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_doxygen_xml = os.path.join(_repo_root, "doxygen-docs", "xml")
_doxygen_xml_index = os.path.join(_doxygen_xml, "index.xml")
breathe_projects = {
    "AnalysisG": _doxygen_xml,
}
breathe_default_project = "AnalysisG"
breathe_default_members = ("members", "undoc-members")

# On Read the Docs the pre_build step runs ``doxygen Doxyfile`` before Sphinx,
# so the XML is always present.  When building locally without having run
# doxygen first, remove breathe from extensions entirely to avoid a hard
# MTimeError exception that would abort the build.
if not os.path.isfile(_doxygen_xml_index):
    extensions = [e for e in extensions if e != "breathe"]
    suppress_warnings = ["app.add_node", "app.add_directive"]
