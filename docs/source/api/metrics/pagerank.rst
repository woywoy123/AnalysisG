PageRank Metric (``metrics.pagerank``)
=======================================

Import with::

    from AnalysisG.metrics.pagerank import PageRankMetric

PageRankMetric
--------------

``PageRankMetric`` is a :class:`~AnalysisG.core.metric_template.MetricTemplate`
subclass that evaluates top-quark reconstruction quality using PageRank-based
node scoring.

It reads three ROOT trees (``event_training``, ``event_validation``,
``event_evaluation``) and for each reads 30 leaves covering truth / PageRank
reconstructed / nominal reconstructed top kinematics plus process mappings.

**``root_leaves`` keys** (full set, set in ``__cinit__``):

.. code-block:: text

    event_training / event_validation / event_evaluation →
        top_truth_{pt,eta,phi,energy,px,py,pz,mass,num_nodes}
        top_pr_reco_{pt,eta,phi,energy,px,py,pz,mass,pagerank,num_nodes}
        top_nom_reco_{pt,eta,phi,energy,px,py,pz,mass,score,num_nodes}
        process_mapping

For each mode the corresponding ``root_fx`` callback
(``get_training`` / ``get_validation`` / ``get_evaluation``) reads a companion
``.txt`` file that maps events to file indices, then calls
``collector.add_file_map`` to build the per-file-per-fold accumulation.
