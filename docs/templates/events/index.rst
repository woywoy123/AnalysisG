Physics Events
=============

The events system in AnalysisG handles the representation of physics events from various experiments and physics processes. The framework includes several event templates found in the ``src/AnalysisG/events/`` directory.

.. toctree::
   :maxdepth: 1
   
   ssml_mc20/index
   bsm_4tops/index
   gnn/index
   exp_mc20/index
   templates/index

Event Model
----------

Events in AnalysisG are handled through the ``event_template`` base class, which defines the interface for all event types. Specific event implementations derive from this template and implement the necessary methods.

.. code-block:: cpp

   // Base event template interface
   class event_template {
      public:
         virtual event_template* clone() = 0;
         virtual void build(element_t* el) = 0;
         virtual void CompileEvent() = 0;
         // ...
   };

Key Event Types
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Event Type
     - Description
   * - ``ssml_mc20``
     - Same-sign multi-lepton events from MC20 campaign
   * - ``bsm_4tops``
     - Beyond Standard Model four top quark events
   * - ``gnn``
     - Graph neural network specialized event structure
   * - ``exp_mc20``
     - Experimental MC20 campaign events

Creating Custom Events
---------------------

To create a custom event type:

1. Inherit from ``event_template``
2. Implement the required virtual methods
3. Register particle types that will be used
4. Define how to build the event from input data
5. Implement the ``CompileEvent`` method to process relationships

Basic Event Template Example
---------------------------

.. code-block:: cpp

   #include "<event-name>.h"

   <event-name>::<event-name>(){
       this->name = "<event-name>"; 
       this->add_leaf("<key_name>", "<leaf-name>"); 
       this->trees = {"<tree-name>"}; 
       this->register_particle(&this->m_particle);
   }

   <event-name>::~<event-name>(){}

   event_template* <event-name>::clone(){return (event_template*)new <event-name>();}

   void <event-name>::build(element_t* el){
       el->get("<key_name>", &this->key_name); 
   }

   void <event-name>::CompileEvent(){
       // Process relationships between particles
   }