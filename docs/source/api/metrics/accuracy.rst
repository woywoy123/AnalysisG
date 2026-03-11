Accuracy Metric (``metrics.accuracy``)
=======================================

Import with::

    from AnalysisG.metrics.accuracy import AccuracyMetric

AccuracyMetric
--------------

``AccuracyMetric`` is a :class:`~AnalysisG.core.metric_template.MetricTemplate`
subclass that computes top-multiplicity classification accuracy over training,
validation, and evaluation modes.

For each epoch and k-fold it reads three ROOT trees: ``event_accuracy_training``,
``event_accuracy_validation``, and ``event_accuracy_evaluation``, each with
leaves ``ntop_truth``, ``ntop_scores``, and ``edge``.

Results
^^^^^^^

After :meth:`Postprocessing`:

* ROC curves for each top-multiplicity class are saved to
  ``./figures/epoch-{N}/{model_name}/ntops.{ext}``.
* Summary AUC-vs-epoch line plots are saved to
  ``./figures/summary/ntop-{class}.{ext}``.
* ``./figures/summary/roc.txt`` contains the AUC table.

**Attributes available after** ``Postprocessing()``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``auc``
     - ``dict[model_name][epoch]`` — AUC results from the ROC curve
       objects for each model at each epoch.

**``root_leaves`` keys** (set in ``__cinit__``):

.. code-block:: python

    {
        "event_accuracy_training":   ["ntop_truth", "ntop_scores", "edge"],
        "event_accuracy_validation": ["ntop_truth", "ntop_scores", "edge"],
        "event_accuracy_evaluation": ["ntop_truth", "ntop_scores", "edge"],
    }

The corresponding ``root_fx`` callbacks populate the internal ``collector``
C++ object which aggregates truth labels and score vectors.
