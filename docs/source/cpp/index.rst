C++ API reference
=================

The C++ API documentation is generated from headers and implementation files via
Doxygen XML and rendered in Sphinx with Breathe.

Primary base/template classes
-----------------------------

- ``tools``
- ``event_template : public tools``
- ``particle_template : public tools``
- ``graph_template : public tools``
- ``selection_template : public tools``
- ``model_template : public notification, public tools``
- ``metric_template : public tools, public notification``
- ``roc : public plotting``

Major derived families
----------------------

- Event implementations derive from ``event_template``:
  ``bsm_4tops``, ``exp_mc20``, ``gnn_event``, ``ssml_mc20``.
- Graph builders derive from ``graph_template`` in
  ``graphs/bsm_4tops``, ``graphs/exp_mc20``, ``graphs/ssml_mc20``.
- Model implementations derive from ``model_template``:
  ``recyclx``, ``grift``, ``recursivegraphneuralnetwork``.
- Metric implementations derive from ``metric_template``:
  ``accuracy_metric``, ``pagerank_metric``, ``topefficiency_metric``.
- Selection implementations derive from ``selection_template`` across
  ``selections/*`` modules (mc16/mc20/analysis/neutrino/performance/example).

Full Doxygen index
------------------

.. doxygenindex::
   :project: AnalysisG

.. note::
   For full API output, generate XML first with ``doxygen docs/Doxyfile``.
