API Reference
=============

This section provides detailed API documentation for all AnalysisG components,
automatically generated from the C++ source code using Doxygen and Breathe.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   core
   events
   graphs
   metrics
   models
   modules
   pyc
   templates

Overview
--------

The AnalysisG API is organized into several main modules:

* :doc:`core` - Core functionality and base templates
* :doc:`events` - Event processing implementations
* :doc:`graphs` - Graph construction from events
* :doc:`metrics` - Performance metrics
* :doc:`models` - Neural network models
* :doc:`modules` - Infrastructure components
* :doc:`pyc` - Python-C++ interface
* :doc:`templates` - Code templates

Namespaces and Classes
----------------------

The framework uses a modular design with clear separation of concerns.
Each module provides specific functionality and can be used independently
or combined for complex analyses.

Documentation Format
--------------------

The API documentation includes:

* **Classes**: Complete class documentation with all members
* **Functions**: Function signatures and descriptions
* **Variables**: Public and private member variables
* **Types**: Type definitions and enumerations
* **Examples**: Usage examples where applicable

Browsing the API
----------------

You can browse the API in several ways:

* Use the table of contents to navigate by module
* Use the search function to find specific classes or functions
* Follow cross-references to related components
* View source code directly from documentation pages

Conventions
-----------

The API follows these conventions:

* **Classes**: CamelCase (e.g., ``event_template``)
* **Functions**: snake_case (e.g., ``build_event``)
* **Member variables**: snake_case with trailing underscore for private (e.g., ``_hidden``)
* **Constants**: UPPER_CASE
