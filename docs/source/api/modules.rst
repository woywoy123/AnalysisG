Modules API
===========

The modules package contains specialized algorithms and functionality built on top of the core framework.

Overview
--------

Modules are located in ``src/AnalysisG/modules/`` and provide the C++ backend implementation for:

- Template classes (event, particle, graph, selection, model, metric)
- Analysis execution engine
- Optimizer algorithms for training
- Plotting utilities beyond core plotting
- Evaluation metrics
- Neutrino reconstruction algorithms
- I/O operations
- Data structures and utilities

All modules integrate seamlessly with the core Analysis framework.

Core Template Implementations
------------------------------

These modules implement the core template classes that users extend.

.. toctree::
   :maxdepth: 2

   modules/analysis
   modules/event
   modules/particle
   modules/graph
   modules/selection

Analysis Components
-------------------

Components for data handling and training.

.. toctree::
   :maxdepth: 2

   modules/optimizer
   modules/lossfx

Specialized Analysis Modules
----------------------------

Modules for specific analysis tasks.

.. toctree::
   :maxdepth: 2

   modules/metrics  
   modules/nusol
   modules/plotting

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

* :doc:`core` - Core API documentation
* :doc:`pyc` - PyC high-performance package

