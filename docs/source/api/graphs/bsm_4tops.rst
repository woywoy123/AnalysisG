BSM 4-Tops Graphs (``graphs.bsm_4tops``)
=========================================

Graph classes for the BSM four-top analysis.  All feature names,
particle node sets, and truth/data labels listed below are derived
directly from the C++ implementation files
``graphs/bsm_4tops/cxx/graphs.cxx``, ``node_features.cxx``,
``edge_features.cxx``, and ``graph_features.cxx``.

Import with::

    from AnalysisG.graphs.bsm_4tops import (
        GraphTops, GraphChildren, GraphTruthJets, GraphTruthJetsNoNu,
        GraphJets, GraphJetsNoNu, GraphDetectorLep, GraphDetector,
    )

All classes inherit :class:`~AnalysisG.core.graph_template.GraphTemplate`
and implement the corresponding C++ graph class from ``<bsm_4tops/graphs.h>``.

Shared Node Features
--------------------

All graph classes that include **non-top particles** register the following
node data features via the ``node_features.cxx`` functions:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Feature name
     - Type
     - Description
   * - ``"pt"``
     - ``double``
     - Transverse momentum :math:`p_T` of the particle
   * - ``"eta"``
     - ``double``
     - Pseudorapidity :math:`\eta`
   * - ``"phi"``
     - ``double``
     - Azimuthal angle :math:`\phi`
   * - ``"energy"``
     - ``double``
     - Energy :math:`E`
   * - ``"charge"``
     - ``double``
     - Electric charge × ``is_lep`` flag (0 for quarks)
   * - ``"is_lep"``
     - ``int``
     - 1 if the particle is a lepton (and not a neutrino), else 0
   * - ``"is_b"``
     - ``int``
     - 1 if the particle is a b-quark, else 0
   * - ``"is_nu"``
     - ``int``
     - 1 if the particle is a neutrino, else 0 *(children graphs only)*

Shared Node Truth Features
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Feature name
     - Type
     - Description
   * - ``"top_node"``
     - ``int``
     - Index of the parent top quark (``-1`` if unmatched)
   * - ``"res_node"``
     - ``int``
     - 1 if the particle originates from the resonance decay, else 0

Shared Edge Truth Features
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Feature name
     - Type
     - Description
   * - ``"res_edge"``
     - ``int``
     - 1 if both endpoints of the edge come from the same resonance
   * - ``"top_edge"``
     - ``int``
     - 1 if both endpoints share at least one common parent top quark

Shared Graph Data Features
--------------------------

All graph classes register the following global graph-level data features:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Feature name
     - Type
     - Description
   * - ``"met"``
     - ``double``
     - Missing transverse energy :math:`E_T^{\rm miss}` from ``event.met``
   * - ``"phi"``
     - ``double``
     - Missing-energy azimuthal angle from ``event.phi``
   * - ``"weight"``
     - ``double``
     - Event weight from ``event.weight``
   * - ``"event_number"``
     - ``long``
     - ROOT event number from ``event.event_number``

Graph Classes
-------------

GraphTops
^^^^^^^^^

Builds one graph per event using **truth top quarks** as nodes
(``event.Tops``).  Topology: all-to-all.

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``

**Node truth:** ``top_node``, ``res_node``

**Edge truth:** ``res_edge``

**Graph data:** ``met``, ``phi``, ``weight``, ``event_number``

**Graph truth:**

.. list-table::
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``"signal"``
     - ``bool``
     - ``True`` if any top quark has ``from_res=True``
   * - ``"ntops"``
     - ``int``
     - Number of top quarks (capped at 4)

GraphChildren
^^^^^^^^^^^^^

Builds one graph per event using **top decay children** as nodes
(``event.Children``).  Topology: all-to-all (``fulltopo``).

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``, ``is_lep``, ``is_b``, ``is_nu``

**Node truth:** ``top_node``, ``res_node``

**Edge truth:** ``res_edge``, ``top_edge``

**Graph data:** ``met``, ``phi``, ``num_jets`` (= number of quarks), ``num_leps``, ``weight``, ``event_number``

**Graph truth:**

.. list-table::
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``"signal"``
     - ``bool``
     - True if any top has ``from_res``
   * - ``"n_nu"``
     - ``int``
     - Number of neutrino children
   * - ``"n_lep"``
     - ``int``
     - Number of charged lepton children
   * - ``"ntops"``
     - ``int``
     - Number of top quarks (capped at 4)

GraphTruthJets
^^^^^^^^^^^^^^

Builds one graph per event from **charged truth leptons + neutrinos + truth jets**
as nodes.  Topology: all-to-all.

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``, ``is_lep``, ``is_b``

**Node truth:** ``top_node``, ``res_node``

**Edge truth:** ``res_edge``, ``top_edge``

**Graph data:** ``met``, ``phi``, ``num_jets`` (= truth-jet count), ``num_leps``, ``weight``, ``event_number``

**Graph truth:** ``signal``, ``n_nu``, ``n_lep``, ``ntops``

GraphTruthJetsNoNu
^^^^^^^^^^^^^^^^^^

Like ``GraphTruthJets`` but **excludes neutrinos** from the node set (only
charged leptons + truth jets).

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``, ``is_lep``, ``is_b``

**Node truth:** ``top_node``, ``res_node``

**Edge truth:** ``res_edge``, ``top_edge``

**Graph data:** ``met``, ``phi``, ``num_jets`` (= truth-jet count), ``num_leps``, ``weight``, ``event_number``

**Graph truth:** ``signal``, ``n_lep``, ``ntops``

GraphJets
^^^^^^^^^

Builds one graph per event using **truth leptons + neutrinos + reconstructed jets**
as nodes.  Pre-selection: exactly 2 detector leptons.  Topology: all-to-all.

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``, ``is_lep``, ``is_b``

**Node truth:** ``top_node``, ``res_node``

**Edge truth:** ``res_edge``, ``top_edge``

**Graph data:** ``met``, ``phi``, ``num_jets`` (= jet count), ``num_leps``, ``weight``, ``event_number``

**Graph truth:** ``signal``, ``n_lep``, ``ntops``

GraphJetsNoNu
^^^^^^^^^^^^^

Like ``GraphJets`` but **excludes neutrino truth children** from the node set.

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``, ``is_lep``, ``is_b``

**Node truth:** ``top_node``, ``res_node``

**Edge truth:** ``res_edge``, ``top_edge``

**Graph data:** ``met``, ``phi``, ``num_jets``, ``num_leps``, ``weight``, ``event_number``

**Graph truth:** ``signal``, ``n_lep``, ``ntops``

GraphDetectorLep
^^^^^^^^^^^^^^^^

Builds one graph per event using **detector muons + detector electrons +
neutrino truth children + reconstructed jets** as nodes.  Topology: all-to-all.

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``, ``is_lep``, ``is_b``

**Node truth:** ``top_node``, ``res_node``

**Edge truth:** ``res_edge``, ``top_edge``

**Graph data:** ``met``, ``phi``, ``num_jets``, ``num_leps`` (= detector lepton count), ``weight``, ``event_number``

**Graph truth:** ``signal``, ``n_lep``, ``ntops``

GraphDetector
^^^^^^^^^^^^^

Builds one graph per event using **all detector objects** (``event.DetectorObjects``).
Pre-selection: exactly 2 detector leptons.  Topology: all-to-all.

**Node data:** ``pt``, ``eta``, ``phi``, ``energy``, ``charge``, ``is_lep``, ``is_b``

**Edge truth:** ``res_edge``, ``top_edge``

**Graph data:** ``met``, ``phi``, ``num_jets``, ``num_leps``, ``weight``, ``event_number``

**Graph truth:** ``signal``, ``n_lep``, ``ntops``

.. note::
   ``GraphDetector`` does **not** register ``top_node`` / ``res_node`` node
   truth features (detector objects do not have direct truth-top parent info).

.. list-table:: GraphDetector-specific properties
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``NumCuda``
     - ``int``
     - Number of CUDA devices to use for graph construction.  Default ``0`` (CPU).
   * - ``ForceMatch``
     - ``bool``
     - When ``True``, require a truth match for each reconstructed object.  Default ``False``.
