.. _api-reference:

API Reference
=============

Complete API documentation generated from C++ and CUDA source code via Doxygen.

Overview
--------

The AnalysisG framework provides comprehensive C++ and CUDA APIs for high energy physics analysis.

For detailed API documentation, please refer to the generated Doxygen HTML documentation available after building the project.

Documentation Structure
-----------------------

The API includes:

* **Events and Particles**: Template classes for event and particle definitions
* **Graphs**: Graph construction utilities for Graph Neural Networks
* **Selections**: Event selection algorithms
* **Metrics**: Evaluation metrics for ML models
* **Models**: Machine learning model implementations
* **PyC**: High-performance C++ and CUDA kernels

Building API Documentation
---------------------------

To generate the complete API documentation locally:

.. code-block:: bash

   # Generate Doxygen documentation
   doxygen Doxyfile
   
   # View the generated HTML documentation
   firefox doxygen-docs/html/index.html
