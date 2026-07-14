Top Efficiency (``selections.performance.topefficiency``)
==========================================================

Import with::

    from AnalysisG.selections.performance.topefficiency import TopEfficiency

TopEfficiency
-------------

``TopEfficiency`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that evaluates GNN model performance for top-quark reconstruction,
including efficiency, purity, invariant-mass distributions, and signal
classification scores.

**Output attributes**:

Invariant-mass distributions (``dict[str, dict[str, list[float]]]``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``p_topmass`` / ``t_topmass``
     - Predicted / truth top-quark invariant mass per decay region.
   * - ``p_zmass`` / ``t_zmass``
     - Predicted / truth Z′ invariant mass per decay region.

Probability scores:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``prob_tops``
     - ``dict[str, dict[str, list[float]]]`` — model score distribution
       for top-quark edges.
   * - ``prob_zprime``
     - ``dict[str, dict[str, list[float]]]`` — model score distribution
       for Z′ edges.
   * - ``t_decay_region`` / ``p_decay_region``
     - Truth / predicted decay-region flag distributions.

Node-count distributions:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``t_nodes`` / ``p_nodes``
     - ``dict[str, dict[float, int]]`` — truth / predicted node-count
       histograms.
   * - ``n_tru_tops``
     - ``dict[str, int]`` — total number of truth top quarks per region.
   * - ``n_pred_tops``
     - ``dict[str, dict[float, int]]`` — predicted top counts per score
       threshold.
   * - ``n_perfect_tops``
     - ``dict[str, dict[float, int]]`` — perfectly reconstructed top
       counts per score threshold.

Per-event score vectors (``list[list[float]]``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``truth_res_edge`` / ``truth_top_edge``
     - ``list[int]`` — per-event truth resonance / top-edge labels.
   * - ``truth_ntops`` / ``truth_signal``
     - ``list[int]`` — per-event truth top multiplicity / signal flag.
   * - ``pred_res_edge_score`` / ``pred_top_edge_score``
     - Per-edge model score vectors for resonance / top classification.
   * - ``pred_ntops_score`` / ``pred_signal_score``
     - Per-event model score vectors for n-top / signal classification.
