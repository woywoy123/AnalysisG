Metrics Module
==============

The Metrics module provides performance metrics for model evaluation.

For complete API reference, see the Doxygen-generated HTML documentation in ``doxygen-docs/html/``.

Metric Implementations
----------------------

Accuracy Metric
~~~~~~~~~~~~~~~

Classification accuracy metrics for model evaluation.

**Location**: ``src/AnalysisG/metrics/accuracy/``

Classes:
* ``accuracy_metric`` - Main accuracy metric implementation
* ``collector`` - Data collection for accuracy analysis
* ``cdata_t`` - Data structure for collected metrics
* ``cmodel_t`` - Model-level metric aggregation

Features:
* Edge-level accuracy
* Node classification accuracy
* K-fold validation support
* Epoch-wise tracking
* Mode-specific metrics

PageRank Metric
~~~~~~~~~~~~~~~

PageRank-based importance scoring for graph nodes.

**Location**: ``src/AnalysisG/metrics/pagerank/``

Features:
* Graph centrality measures
* Node importance ranking
* Iterative PageRank computation
* Weighted graph support

Metric Interface
----------------

All metrics inherit from ``metric_template`` and implement:

* ``define_metric()`` - Metric initialization
* ``define_variables()`` - Variable setup
* ``event()`` - Per-event computation
* ``batch()`` - Per-batch aggregation
* ``end()`` - Final computation

Usage Example
-------------

.. code-block:: cpp

   // Create and configure metric
   auto* metric = new accuracy_metric();
   metric->define_metric(&metric_data);
   metric->define_variables();
   
   // Process events
   for (auto* event : events) {
       metric->event();
   }
   
   // Compute batch metrics
   metric->batch();
   
   // Finalize
   metric->end();
