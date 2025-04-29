========================
AnalysisG Documentation
========================

This documentation was created with Sphinx and is optimized for Read the Docs.

Building the Documentation
--------------------------

To build the documentation locally, execute the following commands:

.. code-block:: bash

    # Install the required packages
    pip install -r docs/requirements.txt

    # Change to the docs directory
    cd docs

    # Build the HTML documentation
    make html

    # Optional: Build the PDF documentation (requires LaTeX)
    # make latexpdf

Directory Structure
-------------------

The documentation is structured as follows:

* ``index.rst``: Main entry point of the documentation
* ``installation.rst``: Installation and configuration guide
* ``quickstart.rst``: Quickstart guide for new users
* ``user_guide/``: Detailed guides for various framework components
* ``api_reference/``: Automatically generated API documentation
* ``tutorials/``: Step-by-step guides for typical tasks
* ``examples/``: Example scripts demonstrating the framework

Extending the Documentation
---------------------------

To extend the documentation:

1. Add new `.rst` files or edit existing ones
2. Update the `toctree` directives in the parent `.rst` files
3. If necessary, update the API documentation with `sphinx-apidoc`
4. Build the documentation with `make html` and check the result


.. _api_reference/index:

==============
API Reference
==============

Details about the AnalysisG classes and functions.

Core Modules
------------

.. toctree::
    :maxdepth: 2

    analysis
    event
    graph
    particle
    dataloader
    optimizer
    model
    io
    meta

C++ Modules and Python Bindings
-------------------------------

AnalysisG is mainly C++ with Python bindings (Cython). Key components:

* **core**: Core classes and functions
* **modules**: Specific functionality modules
* **pyc**: LibTorch and CUDA kernel bindings
* **utils**: Helper functions and tools

Key classes, methods, and functions are documented for each module.


API Reference
============

This section provides detailed documentation for all public classes, methods, and functions in the AnalysisG framework.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   core/index
   modules/index
   events/index
   graphs/index
   selections/index
   pyc/index

Core API
--------

The core API provides the main entry points and base classes for the framework:

.. doxygenclass:: analysis
   :members:
   :undoc-members:

.. doxygenclass:: tools
   :members:
   :undoc-members:

.. doxygenclass:: notification
   :members:
   :undoc-members:

Templates
--------

Templates are the base interfaces for the major components:

.. doxygenclass:: event_template
   :members:
   :undoc-members:

.. doxygenclass:: graph_template
   :members:
   :undoc-members:

.. doxygenclass:: particle_template
   :members:
   :undoc-members:

.. doxygenclass:: selection_template
   :members:
   :undoc-members:

Module Interfaces
---------------

.. doxygenclass:: io
   :members:
   :undoc-members:

.. doxygenclass:: container
   :members:
   :undoc-members:

.. doxygenclass:: meta
   :members:
   :undoc-members:

.. doxygenclass:: plotting
   :members:
   :undoc-members:

Data Structures
-------------

.. doxygenstruct:: settings_t
   :members:
   :undoc-members:

.. doxygenstruct:: data_t
   :members:
   :undoc-members:

.. doxygenstruct:: element_t
   :members:
   :undoc-members:

.. doxygenstruct:: model_report
   :members:
   :undoc-members:

Python Interface
--------------

The Python interface (PyC) exposes the C++ functionality to Python:

.. py:class:: analysisg.Analysis

   Main analysis class for Python interface.
   
   .. py:method:: add_samples(path, label)
      
      Add sample files to the analysis.
      
      :param str path: Path to ROOT files
      :param str label: Label for the sample
      
   .. py:method:: add_event_template(event, label)
      
      Add an event template to the analysis.
      
      :param event_template event: Event template object
      :param str label: Label for the event template
      
   .. py:method:: add_graph_template(graph, label)
      
      Add a graph template to the analysis.
      
      :param graph_template graph: Graph template object
      :param str label: Label for the graph template
      
   .. py:method:: add_selection_template(selection)
      
      Add a selection template to the analysis.
      
      :param selection_template selection: Selection template object
      
   .. py:method:: start()
      
      Start the analysis workflow.



Generating the API Documentation
--------------------------------

To automatically update the API documentation:

.. code-block:: bash

    sphinx-apidoc -o docs/api_reference src/AnalysisG -f