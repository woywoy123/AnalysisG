Metric Interface
================

The Metric Interface provides functionality for implementing evaluation metrics.

Overview
--------

Metrics are used to evaluate model performance. The MetricTemplate class supports:

* Custom metric definitions
* State management across batches
* Reset and update operations
* Integration with training loops

Core MetricTemplate Class
-------------------------

File Location
~~~~~~~~~~~~~

* **Cython Implementation**: ``src/AnalysisG/core/metric_template.pyx``
* **Cython Header**: ``src/AnalysisG/core/metric_template.pxd``

Methods to Override
~~~~~~~~~~~~~~~~~~~

.. method:: compute()
   
   Calculate the metric value.

.. method:: update()
   
   Update metric state with new predictions.

.. method:: reset()
   
   Reset metric state.

See Also
--------

* :doc:`../core/metric_template`: Core MetricTemplate implementation
* :doc:`../metrics/overview`: Concrete metric implementations
