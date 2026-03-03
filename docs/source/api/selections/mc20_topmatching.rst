MC20 Top-Quark Matching (``selections.mc20.topmatching``)
==========================================================

Import with::

    from AnalysisG.selections.mc20.topmatching import TopMatching

TopMatching
-----------

``TopMatching`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that matches truth top quarks to reconstructed jets and leptons for MC20
samples.

**Output attributes**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``truth_top``
     - ``list[float]`` — per-event truth top-quark invariant mass.
   * - ``no_children``
     - ``list[int]`` — number of top quarks with no matched children.
   * - ``truth_children``
     - ``dict[str, list[float]]`` — matched truth-children kinematics.
   * - ``truth_jets``
     - ``dict[str, list[float]]`` — matched truth-jet kinematics.
   * - ``n_truth_jets_lep``
     - ``dict[str, list[float]]`` — truth jets matched to leptonic tops.
   * - ``n_truth_jets_had``
     - ``dict[str, list[float]]`` — truth jets matched to hadronic tops.
   * - ``jets_truth_leps``
     - ``dict[str, list[float]]`` — truth jets associated with leptons.
   * - ``jet_leps``
     - ``dict[str, list[float]]`` — reco jets associated with leptons.
   * - ``n_jets_lep``
     - ``dict[str, list[float]]`` — reco jets per leptonic top.
   * - ``n_jets_had``
     - ``dict[str, list[float]]`` — reco jets per hadronic top.
