MetricTemplate (Python)
=======================

The ``MetricTemplate`` Cython class wraps the C++ ``metric_template``.
User metric classes must subclass it and override ``Postprocessing``.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``RunNames``
     - ``dict[str, str]``
     - Map of run-name strings used to group metric results.
   * - ``Variables``
     - ``list[str]``
     - Feature names to evaluate.

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``Postprocessing()``
     - Override in subclasses to compute and plot aggregate metrics.
   * - ``InterpretROOT(path: str, epochs: list = [], kfolds: list = [])``
     - Re-process saved ROOT output files to repopulate metrics without
       re-running the full training pipeline.
