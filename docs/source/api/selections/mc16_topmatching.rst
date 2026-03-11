MC16 Top-Quark Matching (``selections.mc16.topmatching``)
==========================================================

Import with::

    from AnalysisG.selections.mc16.topmatching import TopMatching

TopMatching
-----------

``TopMatching`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that matches truth top quarks to their jets and leptons in MC16 samples.

**Output attributes**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``truth_top``
     - ``list`` — truth top-quark mass values.
   * - ``no_children``
     - ``list`` — list of top quarks for which no decay products were found.
   * - ``truth_children``
     - ``dict[str, list]`` — matched truth-children objects.
   * - ``truth_jets``
     - ``dict[str, list]`` — matched truth-jet objects.
   * - ``n_truth_jets_lep``
     - ``dict[str, list]`` — number of truth jets matched to leptonic tops.
   * - ``n_truth_jets_had``
     - ``dict[str, list]`` — number of truth jets matched to hadronic tops.
   * - ``jets_truth_leps``
     - ``dict[str, list]`` — truth jets associated with leptonic decays.
   * - ``jet_leps``
     - ``dict[str, list]`` — jets associated with leptons.
   * - ``n_jets_lep``
     - ``dict[str, list]`` — number of reco jets per leptonic top.
   * - ``n_jets_had``
     - ``dict[str, list]`` — number of reco jets per hadronic top.
