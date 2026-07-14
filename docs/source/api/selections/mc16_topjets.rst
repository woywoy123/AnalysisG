MC16 Top + Reconstructed-Jet Matching (``selections.mc16.topjets``)
====================================================================

Import with::

    from AnalysisG.selections.mc16.topjets import TopJets

TopJets
-------

``TopJets`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that matches truth top quarks to reconstructed jets and studies their
parton contributions in MC16 samples.

**Output attributes**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``top_mass``
     - ``dict[str, dict[str, dict[str, list[float]]]]`` — invariant mass
       of top-quark candidates reconstructed from reco jets.
   * - ``jet_partons``
     - Nested dict of parton-level contributions to reconstructed jets.
   * - ``jets_contribute``
     - Nested dict of per-top jet contributions.
   * - ``jet_top``
     - ``dict[str, dict[str, list[float]]]`` — jet-to-top matching
       metrics.
   * - ``jet_mass``
     - ``dict[str, list[float]]`` — reconstructed jet invariant masses.
   * - ``ntops_lost``
     - ``list[int]`` — number of top quarks with no matched reco jet.
