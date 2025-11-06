Model Interface
===============

The Model Interface provides functionality for implementing machine learning models.

Overview
--------

The ModelTemplate class provides:

* Model architecture definition
* Forward pass implementation
* Loss function definition
* Integration with training framework

Core ModelTemplate Class
------------------------

File Location
~~~~~~~~~~~~~

* **Cython Implementation**: ``src/AnalysisG/core/model_template.pyx``
* **Cython Header**: ``src/AnalysisG/core/model_template.pxd``

Methods to Override
~~~~~~~~~~~~~~~~~~~

.. method:: forward()
   
   Model forward pass.

.. method:: loss()
   
   Loss function computation.

.. method:: predict()
   
   Generate predictions.

See Also
--------

* :doc:`../core/model_template`: Core ModelTemplate implementation
* :doc:`../models/overview`: Concrete model implementations
