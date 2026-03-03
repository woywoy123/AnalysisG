BSM 4-Tops Graphs (``graphs.bsm_4tops``)
=========================================

Graph classes for the BSM four-top analysis.  Import with::

    from AnalysisG.graphs.bsm_4tops import (
        GraphTops, GraphChildren, GraphTruthJets, GraphTruthJetsNoNu,
        GraphJets, GraphJetsNoNu, GraphDetectorLep, GraphDetector,
    )

All classes inherit :class:`~AnalysisG.core.graph_template.GraphTemplate`
and implement the corresponding C++ graph class from ``<bsm_4tops/graphs.h>``.

Graph Classes
-------------

GraphTops
^^^^^^^^^

Builds one graph per event using **truth top quarks** as nodes.
No additional properties beyond the base class.

GraphChildren
^^^^^^^^^^^^^

Builds one graph per event using **top decay children** as nodes.
No additional properties beyond the base class.

GraphTruthJets
^^^^^^^^^^^^^^

Builds one graph per event using **truth jets** (including neutrino
truth products) as nodes.
No additional properties beyond the base class.

GraphTruthJetsNoNu
^^^^^^^^^^^^^^^^^^

Builds one graph per event using **truth jets**, **excluding neutrinos**
from the node set.
No additional properties beyond the base class.

GraphJets
^^^^^^^^^

Builds one graph per event using **reconstructed jets** (and leptons) as
nodes.
No additional properties beyond the base class.

GraphJetsNoNu
^^^^^^^^^^^^^

Builds one graph per event using **reconstructed jets**, **excluding
neutrino candidates** from the node set.
No additional properties beyond the base class.

GraphDetectorLep
^^^^^^^^^^^^^^^^

Builds one graph per event using **detector-level objects**, including
lepton objects.
No additional properties beyond the base class.

GraphDetector
^^^^^^^^^^^^^

Builds one graph per event using the full **detector-level** object
collection, optionally forcing truth-matching and running on multiple CUDA
devices.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``NumCuda``
     - ``int``
     - Number of CUDA devices to use for graph construction.  Default ``0``
       (CPU).
   * - ``ForceMatch``
     - ``bool``
     - When ``True``, require a truth match for each reconstructed object
       before adding it to the graph.  Default ``False``.
