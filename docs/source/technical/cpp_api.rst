C++ API Reference
==================

This section documents the C++ API for AnalysisG.

Overview
--------

The C++ API provides high-performance implementations of core algorithms.

Using Doxygen Output
--------------------

C++ documentation is generated using Doxygen and integrated via Breathe.

To view C++ class documentation, see the modules section:

* :doc:`../modules/overview`

Building Doxygen Documentation
-------------------------------

.. code-block:: bash

   cd docs
   doxygen Doxyfile

This generates XML output in ``docs/xml/`` which is consumed by Breathe.

See Also
--------

* :doc:`overview`: Technical overview
* :doc:`cuda_api`: CUDA API reference
