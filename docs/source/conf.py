from __future__ import annotations

from pathlib import Path

project = "AnalysisG"
author = "AnalysisG contributors"

extensions = [
    "breathe",
    "sphinx.ext.graphviz",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"

ROOT = Path(__file__).resolve().parents[2]
DOXYGEN_XML = ROOT / "docs" / "build" / "doxygen" / "xml"

breathe_projects = {"AnalysisG": str(DOXYGEN_XML)}
breathe_default_project = "AnalysisG"
breathe_default_members = ("members", "undoc-members")

if not DOXYGEN_XML.exists():
    DOXYGEN_XML.mkdir(parents=True, exist_ok=True)
    (DOXYGEN_XML / "index.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?><doxygenindex version='1.9.8'/>",
        encoding="utf-8",
    )

graphviz_output_format = "svg"
