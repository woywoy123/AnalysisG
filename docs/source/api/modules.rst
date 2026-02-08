Modules API
===========

The modules package contains specialized algorithms and functionality built on top of the core framework.

Overview
--------

Modules are located in ``src/AnalysisG/modules/`` and provide:

- Optimizer algorithms for training
- Plotting utilities beyond core plotting
- Evaluation metrics
- Neutrino reconstruction algorithms

All modules integrate seamlessly with the core Analysis framework.

Available Modules
-----------------

.. toctree::
   :maxdepth: 1

   optimizer
   metrics  
   nusol
   plotting

Module Structure
----------------

Each module typically contains:

- C++ implementation (``cxx/`` directory)
- C++ headers (``include/`` directory)
- Python/Cython bindings (``__init__.py`` or ``.pyx`` files)
- Documentation

Modules follow the same template pattern as core components, providing:

- Base classes to extend
- Configuration options
- Integration with Analysis class

See Also
--------

* :doc:`../core` - Core API documentation
* :doc:`../pyc` - PyC high-performance package
