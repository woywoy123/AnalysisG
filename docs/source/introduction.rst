Introduction
============

Overview
--------

AnalysisG is a high-performance Graph Neural Network (GNN) framework designed
specifically for High Energy Physics (HEP) analysis. It combines the power of
C++ for computational efficiency with the flexibility of Python for ease of use.

What is AnalysisG?
------------------

AnalysisG provides a complete ecosystem for:

* **Data Processing**: Read and process physics events from ROOT files
* **Graph Construction**: Transform physics events into graph representations
* **Neural Networks**: Train and deploy graph neural networks
* **Analysis**: Perform physics analysis with built-in tools
* **Visualization**: Create plots and visualizations of results

Framework Architecture
----------------------

The framework is organized into several key components:

Core Components
~~~~~~~~~~~~~~~

* **Core Module**: Base templates and fundamental functionality
* **Events**: Event processing for various physics analyses
* **Graphs**: Graph construction from physics events
* **Models**: Neural network model implementations
* **Metrics**: Performance evaluation and metrics

Infrastructure
~~~~~~~~~~~~~~

* **Modules**: Low-level infrastructure components
* **PyC**: Python-C++ interface layer via Cython
* **Templates**: Code templates for custom implementations

Design Philosophy
-----------------

AnalysisG follows these design principles:

1. **Performance**: C++ implementation for computational efficiency
2. **Usability**: Python interface for ease of use
3. **Extensibility**: Template-based architecture for customization
4. **Modularity**: Independent, reusable components
5. **Flexibility**: Support for various analysis workflows

Use Cases
---------

AnalysisG is suitable for:

* Top quark pair production analysis
* Beyond Standard Model (BSM) searches
* Multi-lepton final states
* Jet reconstruction and classification
* Graph-based event classification
* Neural network model development for HEP

Technology Stack
----------------

The framework is built on:

* **C++17**: Core implementation language
* **Cython**: Python-C++ interface
* **PyTorch/LibTorch**: Neural network framework
* **ROOT**: High-energy physics data format
* **CMake**: Build system
* **Doxygen**: C++ documentation
* **Sphinx**: Python and overall documentation

Getting Help
------------

For questions and support:

* Check the API documentation in :doc:`api/index`
* Review the module documentation in :doc:`modules/index`
* Look at examples in :doc:`examples`
* Visit the GitHub repository for issues and discussions

Next Steps
----------

Continue to:

* :doc:`api/index` - Detailed API documentation
* :doc:`modules/index` - Module-by-module documentation
* :doc:`examples` - Usage examples and tutorials
