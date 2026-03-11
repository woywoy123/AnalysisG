Same-Sign Multi-Lepton MC20 Graphs (``graphs.ssml_mc20``)
==========================================================

Graph classes for the same-sign multi-lepton MC20 analysis.  Import
with::

    from AnalysisG.graphs.ssml_mc20 import (
        GraphJets, GraphJetsNoNu, GraphDetectorLep, GraphDetector,
    )

All classes inherit :class:`~AnalysisG.core.graph_template.GraphTemplate`
and wrap the corresponding C++ graph class from ``<ssml_mc20/graphs.h>``.

Graph Classes
-------------

GraphJets
^^^^^^^^^

Builds one graph per event using **reconstructed jets** (and associated
leptons) as nodes.

GraphJetsNoNu
^^^^^^^^^^^^^

Builds one graph per event using **reconstructed jets**, excluding neutrino
candidates.

GraphDetectorLep
^^^^^^^^^^^^^^^^

Builds one graph per event using **detector-level objects**, including
lepton objects.

GraphDetector
^^^^^^^^^^^^^

Builds one graph per event using the full **detector-level** collection.
