Selection Interface
===================

The Selection Interface provides functionality for implementing selection criteria.

Overview
--------

The SelectionTemplate class provides:

* Event selection logic
* Object selection criteria
* Cut flow management

Core SelectionTemplate Class
-----------------------------

File Location
~~~~~~~~~~~~~

* **Cython Implementation**: ``src/AnalysisG/core/selection_template.pyx``
* **Cython Header**: ``src/AnalysisG/core/selection_template.pxd``

Methods to Override
~~~~~~~~~~~~~~~~~~~

.. method:: apply()
   
   Apply selection to events.

.. method:: passes()
   
   Check if object passes selection.

See Also
--------

* :doc:`../core/selection_template`: Core SelectionTemplate implementation
