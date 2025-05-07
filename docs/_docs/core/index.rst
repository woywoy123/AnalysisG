Core Framework
=============

The core of AnalysisG is built around the ``analysis`` class, which serves as the central entry point for all physics analysis operations.

Analysis Class
-------------

.. code-block:: cpp

   class analysis: 
       public notification, 
       public tools
   {
       // ...
   };

The Analysis class coordinates all aspects of a physics analysis workflow:

* Data input/output management
* Event template registration and processing
* Graph generation from events
* Selection application
* Model training and inference
* Output generation

Key Operations
-------------

.. list-table::
   :header-rows: 1

   * - Method
     - Purpose
   * - ``add_samples()``
     - Add ROOT file samples to the analysis with a label
   * - ``add_event_template()``
     - Register an event template for processing data
   * - ``add_graph_template()``
     - Register a graph template for creating graph structures
   * - ``add_selection_template()``
     - Register a selection to be applied to events
   * - ``add_model()``
     - Register a model for training or inference
   * - ``start()``
     - Begin the analysis workflow
   * - ``build_events()``
     - Process ROOT files into event objects
   * - ``build_graphs()``
     - Generate graph structures from events

Analysis Flow
------------

A typical analysis workflow consists of:

1. Creating an Analysis object
2. Registering event templates, graph templates, and selections
3. Adding input samples
4. Starting the analysis
5. Retrieving and visualizing results

Example
-------

.. code-block:: cpp

   // Create an analysis
   analysis* ana = new analysis();
   
   // Configure input samples
   ana->add_samples("/path/to/ttbar_samples/*.root", "ttbar");
   ana->add_samples("/path/to/wjets_samples/*.root", "wjets");
   
   // Set up event processing
   ana->add_event_template(new ssml_mc20(), "events");
   
   // Create graph representation
   ana->add_graph_template(new graph_jets(), "jets");
   
   // Apply selection regions
   ana->add_selection_template(new regions());
   
   // Start analysis
   ana->start();

Key Components
-------------

.. toctree::
   :maxdepth: 1
   
   tools
   notification
   templates