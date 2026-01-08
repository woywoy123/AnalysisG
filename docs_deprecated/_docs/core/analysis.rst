Analysis Module
==============

The Analysis module is the central orchestration component of the AnalysisG framework, responsible for coordinating all aspects of a physics analysis workflow.

Overview
--------

The Analysis class serves as the entry point for most operations in the framework, allowing you to:

* Add input data samples
* Register event templates for processing data
* Register graph templates for creating graph representations
* Apply selections to events
* Configure and run machine learning models
* Process and analyze data in parallel

Key Features
-----------

* **Workflow Orchestration**: Coordinates the flow of data through all framework components
* **Parallel Processing**: Handles multi-threaded execution for improved performance
* **Data Management**: Manages input data, event generation, and cached results
* **Model Coordination**: Sets up model training, evaluation, and inference

Core Methods
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``add_samples()``
     - Add ROOT file samples to the analysis with a label
   * - ``add_event_template()``
     - Register an event template to process event data
   * - ``add_graph_template()``
     - Register a graph template to create graph representations
   * - ``add_selection_template()``
     - Register a selection to filter events
   * - ``add_model()``
     - Register a machine learning model for training or inference
   * - ``add_metric_template()``
     - Add a metric for model evaluation
   * - ``start()``
     - Begin the analysis workflow
   * - ``build_events()``
     - Process ROOT files into event objects
   * - ``build_graphs()``
     - Generate graph structures from events

Implementation Details
--------------------

.. code-block:: cpp

   class analysis: public notification, public tools {
   public:
       analysis();
       ~analysis();

       void add_samples(std::string path, std::string label);
       void add_event_template(event_template* ev, std::string label);
       void add_graph_template(graph_template* gr, std::string label);
       void add_selection_template(selection_template* sel);
       void add_metric_template(metric_template* mx, model_template* mdl);
       void add_model(model_template* model, optimizer_params_t* op, std::string run_name);
       void add_model(model_template* model, std::string run_name);
       void start();
       
       // Configuration settings
       settings_t m_settings;
   };

Example Usage
------------

.. code-block:: cpp

   // Create analysis object
   analysis* ana = new analysis();
   
   // Add data samples
   ana->add_samples("/path/to/data/*.root", "ttbar");
   
   // Register event template
   ana->add_event_template(new ssml_mc20(), "events");
   
   // Register graph template
   ana->add_graph_template(new graph_jets(), "jets");
   
   // Register selection
   ana->add_selection_template(new regions());
   
   // Start analysis
   ana->start();

Internal Components
-----------------

The Analysis module coordinates several internal components:

* **Event Building**: Creates physics event objects from ROOT files
* **Graph Building**: Creates graph representations from physics events
* **Selection Application**: Applies filters to events based on physics criteria
* **Model Training**: Manages the training of machine learning models
* **Inference**: Runs prediction on new data with trained models