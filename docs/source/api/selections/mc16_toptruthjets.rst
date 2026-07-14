MC16 Top + Truth-Jet Matching (``selections.mc16.toptruthjets``)
=================================================================

Import with::

    from AnalysisG.selections.mc16.toptruthjets import TopTruthJets

TopTruthJets
------------

``TopTruthJets`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that matches truth top quarks to truth jets and their constituent partons
in MC16 samples.

**Output attributes**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``top_mass``
     - ``dict[str, dict[str, dict[str, list[float]]]]`` — invariant masses
       of top-quark candidates reconstructed from truth jets.
   * - ``truthjet_partons``
     - Nested dict of parton-level contributions to truth jets.
   * - ``truthjets_contribute``
     - Nested dict of per-top truth-jet contributions.
   * - ``truthjet_top``
     - ``dict[str, dict[str, list[float]]]`` — truth-jet-to-top matching
       metrics.
   * - ``truthjet_mass``
     - ``dict[str, list[float]]`` — truth-jet invariant masses.
   * - ``ntops_lost``
     - ``list[int]`` — number of top quarks with no matched truth jet.
