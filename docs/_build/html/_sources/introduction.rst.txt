Introduction to AnalysisG
=========================

Overview
--------

AnalysisG is a comprehensive Graph Neural Network Analysis Framework specifically designed for High Energy Physics (HEPP) applications. 
The framework addresses the computational challenges faced when converting ROOT-based data structures into formats suitable for machine 
learning algorithms, particularly Graph Neural Networks.

Philosophy
----------

AnalysisG follows a philosophy similar to *AnalysisTop*, treating events and particles as polymorphic objects. This approach provides:

* **Flexibility**: Easy adaptation to different analyses and data formats
* **Performance**: C++ backend with Cython bindings for optimal speed
* **Simplicity**: Intuitive Python API that hides implementation complexity
* **Extensibility**: Template-based design for custom implementations

Architecture
------------

The framework is built on several key architectural principles:

Template-Based Design
~~~~~~~~~~~~~~~~~~~~~

AnalysisG uses template classes that users extend to define their specific analysis:

* **EventTemplate**: Define event-level structures and behavior
* **ParticleTemplate**: Define particle-level structures and behavior
* **GraphTemplate**: Define graph structures for GNN applications
* **SelectionTemplate**: Define event selection criteria
* **MetricTemplate**: Define evaluation metrics

Polymorphic Objects
~~~~~~~~~~~~~~~~~~~

Events and particles are treated as polymorphic objects, allowing:

* Natural representation of complex decay chains
* Easy truth matching studies
* Flexible particle relationship definitions
* Efficient memory management

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

The framework achieves high performance through:

* **C++ Core**: Performance-critical code written in C++
* **Cython Interface**: Minimal overhead Python bindings
* **CUDA Kernels**: Native GPU acceleration for key operations
* **Efficient Memory**: Shared pointers and minimal copying

Core Components
---------------

Core Module
~~~~~~~~~~~

The core module (``src/AnalysisG/core``) contains the fundamental building blocks:

* ``analysis.pyx``: Main analysis compilation and execution
* ``event_template.pyx``: Base event template class
* ``particle_template.pyx``: Base particle template class
* ``graph_template.pyx``: Base graph template class
* ``selection_template.pyx``: Base selection template class
* ``metric_template.pyx``: Base metric template class
* ``model_template.pyx``: Base model template class
* ``plotting.pyx``: Plotting utilities
* ``io.pyx``: ROOT I/O interface
* ``tools.pyx``: Utility functions

Modules
~~~~~~~

The modules directory (``src/AnalysisG/modules``) contains specialized functionality:

* **optimizer**: Training optimization algorithms
* **plotting**: Advanced plotting routines
* **metrics**: Evaluation metrics implementations
* **nusol**: Neutrino reconstruction algorithms

PyC Package
~~~~~~~~~~~

The PyC package (``pyc/pyc``) is a self-contained high-performance package:

* **physics**: Physics calculations (ΔR, invariant mass, etc.)
* **operators**: Tensor operations
* **graph**: Graph operations
* **transform**: Coordinate transformations
* **nusol**: Neutrino reconstruction
* **plotting**: Fast plotting utilities
* **tools**: Utility functions

Use Cases
---------

AnalysisG is designed for:

1. **Graph Neural Network Training**: Convert HEPP data to graph structures
2. **Truth Matching Studies**: Match reconstructed particles to truth information
3. **Cut-Based Analyses**: Implement selection criteria and export results
4. **Neutrino Reconstruction**: Analytical and numerical neutrino reconstruction
5. **Performance Studies**: Evaluate algorithm efficiency and accuracy

Analysis Agnostic
-----------------

The framework aims to remain analysis-agnostic, allowing:

* Multiple ATLAS analyses to benefit from shared infrastructure
* Easy adaptation to other experiments (CMS, LHCb, etc.)
* Flexible integration with existing analysis workflows
* Community-driven development and contributions

Next Steps
----------

* Check out the :doc:`installation` guide to get started
* Explore the :doc:`api/index` for detailed API documentation
* See :doc:`examples/index` for practical usage examples
