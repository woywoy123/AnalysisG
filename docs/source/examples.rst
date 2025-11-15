Examples
========

This page provides usage examples for the AnalysisG framework.

Basic Event Processing
-----------------------

Creating and Processing Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <event_template.h>
   
   // Create event instance
   auto* event = new MyEvent();
   
   // Configure event
   event->trees = {"nominal"};
   event->branches = {"Jets", "Leptons"};
   
   // Build from data
   event->build(element);
   
   // Process event
   event->CompileEvent();
   
   // Access particles
   for (auto* particle : event->particles) {
       std::cout << "pT: " << particle->pt << std::endl;
   }

Graph Construction
------------------

Building Graphs from Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <graph_template.h>
   
   // Create graph
   auto* graph = new MyGraph();
   
   // Build from event
   graph->build(event);
   
   // Access graph components
   auto nodes = graph->GetNodes();
   auto edges = graph->GetEdges();
   auto features = graph->GetFeatures();

Model Training
--------------

Training a GNN Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <models/grift.h>
   
   // Create model
   auto* model = new grift();
   
   // Configure model
   model->_hidden = 512;
   model->drop_out = 0.1;
   
   // Forward pass
   model->forward(&graph_data);
   
   // Get predictions
   auto predictions = graph_data.edge_predictions;

Python Interface
----------------

Using from Python
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG import Event, Graph, Model
   
   # Load data
   event = Event()
   event.load_from_file("data.root")
   
   # Build graph
   graph = Graph()
   graph.build(event)
   
   # Run model
   model = Model.load("model.pt")
   predictions = model(graph)

Physics Calculations
--------------------

Computing Physics Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG import PhysicsUtils
   
   # Delta R
   dr = PhysicsUtils.delta_r(particle1, particle2)
   
   # Invariant mass
   mass = PhysicsUtils.invariant_mass([p1, p2, p3])
   
   # Transverse momentum
   pt = particle.pt

Custom Metrics
--------------

Implementing Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   class MyMetric : public metric_template {
   public:
       void define_metric(metric_t* mtx) override {
           // Initialize metric
       }
       
       void event() override {
           // Process each event
           compute_event_metric();
       }
       
       void batch() override {
           // Aggregate batch results
           aggregate_results();
       }
       
       void end() override {
           // Finalize metric
           compute_final_metric();
       }
   
   private:
       void compute_event_metric();
       void aggregate_results();
       void compute_final_metric();
   };

More Examples
-------------

For more examples, see:

* Template implementations in ``src/AnalysisG/templates/``
* Test files in ``test/`` directory
* Study implementations in ``studies/`` directory
* Production scripts in ``sample-production/``

See Also
--------

* :doc:`api/index` - API reference
* :doc:`modules/index` - Module documentation
* :doc:`introduction` - Framework introduction
