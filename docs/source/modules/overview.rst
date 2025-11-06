Modules Package Overview
========================

The modules package contains the C++ implementation of core AnalysisG functionality.

Overview
--------

This package consists of 23 submodules, each providing specific functionality:

* **analysis**: Main analysis framework
* **container**: Data containers and storage
* **dataloader**: Data loading infrastructure
* **event**: Event processing
* **graph**: Graph data structures
* **io**: Input/output operations
* **lossfx**: Loss function implementations
* **meta**: Metadata handling
* **metric**: Metric calculations
* **metrics**: Additional metrics
* **model**: Model implementations
* **notification**: Notification system
* **nusol**: Neutrino reconstruction
* **optimizer**: Optimization algorithms
* **particle**: Particle handling
* **plotting**: Plotting utilities
* **roc**: ROC curve calculations
* **sampletracer**: Sample tracking
* **selection**: Selection logic
* **structs**: Core data structures
* **tools**: Utility tools
* **typecasting**: Type conversion utilities

All modules are implemented in C++ for maximum performance.

File Organization
-----------------

Each module follows this structure:

.. code-block:: text

   module_name/
   ├── CMakeLists.txt
   ├── cxx/
   │   └── *.cxx files
   └── include/
       └── *.h files

See Also
--------

* :doc:`../technical/overview`: Technical overview
* :doc:`../technical/cpp_api`: C++ API reference
