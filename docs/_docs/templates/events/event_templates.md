.. _event_templates:

=================
Event Templates
=================

The `event_template` is a core class in AnalysisG used to define high-energy physics events.

Basic Concept
-------------

An event object represents a single physics event and contains:

*   **Event Variables**: Global information like Missing ET, weights, etc.
*   **Particle Collections**: Groups of particles (e.g., jets, electrons, muons).
*   **Metadata**: Additional event information.

Implementing an Event Template
------------------------------

Here's an example of defining an event template class:

.. code-block:: cpp

    class TTbarEvent : public event_template
    {
    public:
         TTbarEvent()
         {
              this->name = "ttbar_event";

              // Define event variables
              this->add_leaf("met", "MET_met");
              this->add_leaf("met_phi", "MET_phi");
              this->add_leaf("eventNumber", "eventNumber");

              // Create particle objects
              this->jets = new Jet();
              this->electrons = new Electron();
              this->muons = new Muon();

              // Register particle collections
              this->register_particle(this->jets);
              this->register_particle(this->electrons);
              this->register_particle(this->muons);
         }

         ~TTbarEvent() {}

         event_template* clone() override
         {
              return (event_template*)new TTbarEvent();
         }

         void CompileEvent() override
         {
              // Additional event-level calculations
              // e.g., compute derived variables
         }

         // Access to particle collections
         Jet* jets;
         Electron* electrons;
         Muon* muons;
    };

Accessing ROOT Data
-------------------

Event templates automatically connect to ROOT TTrees to read data:

.. code-block:: cpp

    // In your event template class:
    this->add_leaf("variable_name", "root_leaf_name");

    // This allows accessing the ROOT leaf "root_leaf_name" via:
    float value = this->variable_name;

Particle Registration
---------------------

Register different types of particles within your event:

.. code-block:: cpp

    // Register a single particle
    Jet* jet = new Jet();
    this->register_particle(jet);

    // Register a collection of particles with a size variable
    Jets* jets = new Jets();
    this->register_particle(jets, "n_jets");

    // The size is automatically read from the ROOT leaf "n_jets"

Processing Events
-----------------

The framework reads events from ROOT files automatically:

.. code-block:: python

    # In Python:
    from AnalysisG.core import analysis

    # Create analysis object
    my_analysis = analysis()

    # Add input files
    my_analysis.add_file("input.root", "data")

    # Register event template
    my_analysis.register_event("TTbar", TTbarEvent)

    # Start analysis (reads events)
    my_analysis.start()

Relation to Other Components
----------------------------

Event templates interact with other framework components:

*   **Particle Templates**: Define the particles within the event.
*   **Graph Templates**: Use event data to build graphs.
*   **IO System**: Reads event data from ROOT files.
*   **Analysis**: Coordinates event processing.

Further Resources
-----------------

*   Full API documentation: :ref:`API-Event-Template <api_reference/event>`
*   Example code in the `/docs/examples` directory.