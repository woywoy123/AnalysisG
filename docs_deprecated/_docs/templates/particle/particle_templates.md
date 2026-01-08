.. _particle_templates:

==================
Particle Templates
==================

The `particle_template` serves as the foundation for defining particles within the AnalysisG framework.

Core Concept
------------

A `Particle` object represents an individual particle (e.g., jet, electron, muon) within a single event. It typically holds:

*   **Kinematics**: Transverse momentum (pT), pseudorapidity (eta), azimuthal angle (phi), energy, mass, etc.
*   **Identification**: Flavor tags, charge, quality flags, etc.
*   **Calibration**: Associated scale factors, efficiencies, uncertainties, etc.

Implementing a Particle Template
--------------------------------

To define a custom particle type, inherit from `particle_template`. Here's an example for a `Jet`:

.. code-block:: cpp

     #include "AnalysisG/Templates/particle_template.h" // Assuming header location
     #include <cmath> // For std::sqrt

     class Jet : public particle_template
     {
     public:
           Jet()
           {
                 // --- Kinematic Variables ---
                 // Map class members to ROOT TTree leaf names
                 this->add_leaf("pt", "jet_pt");
                 this->add_leaf("eta", "jet_eta");
                 this->add_leaf("phi", "jet_phi");
                 this->add_leaf("energy", "jet_energy");
                 // Assuming mass might be needed for mt() or other calculations
                 // this->add_leaf("mass", "jet_mass");

                 // --- Jet-Specific Properties ---
                 this->add_leaf("btag_score", "jet_btag_score");
                 this->add_leaf("jvt", "jet_jvt"); // Jet Vertex Tagger
                 this->add_leaf("width", "jet_width");

                 // --- Monte Carlo Truth Information (Optional) ---
                 this->add_leaf("true_flavor", "jet_true_flavor");
           }

           // --- Calculated Properties ---
           // Example: Calculate transverse mass (definition might vary)
           // Note: Ensure 'mass' is defined and linked if used here.
           // float mt() const
           // {
           //      // Example calculation: E_T = sqrt(pt^2 + mass^2)
           //      // return std::sqrt(this->pt * this->pt + this->mass * this->mass);
           //      // Or if mass is negligible: return this->pt;
           //      return this->pt; // Placeholder if mass is not readily available/needed
           // }

           // --- Member Variables (automatically managed by particle_template) ---
           // float pt = 0;
           // float eta = 0;
           // float phi = 0;
           // float energy = 0;
           // float mass = 0; // Add if needed
           // float btag_score = 0;
           // float jvt = 0;
           // float width = 0;
           // int true_flavor = 0; // Assuming integer type
     };

Particle Collections
--------------------

To handle multiple particles of the same type within an event (e.g., several jets), define a collection class that inherits from the single particle template:

.. code-block:: cpp

     #include <vector>
     #include <cstddef> // For size_t

     class Jets : public Jet // Inherits Jet properties and leaf mappings
     {
     public:
           Jets() : Jet() {} // Call base constructor

           // --- Optional: Collection-Specific Methods ---
           // Example: Get a vector of b-tagged jets above a certain score threshold
           std::vector<Jet> get_bjets(float threshold = 0.7) const
           {
                 std::vector<Jet> bjets;
                 // 'this->size()' gives the number of jets in the current event
                 for (size_t i = 0; i < this->size(); ++i)
                 {
                         // Access properties of the i-th jet using array-like access
                         if (this->btag_score[i] > threshold)
                         {
                               // Create a temporary Jet object for the collection
                               Jet current_jet;
                               // Manually copy properties for the i-th jet
                               // Note: particle_template might offer a more direct way
                               //       to get a single particle instance by index.
                               current_jet.pt = this->pt[i];
                               current_jet.eta = this->eta[i];
                               current_jet.phi = this->phi[i];
                               current_jet.energy = this->energy[i];
                               current_jet.btag_score = this->btag_score[i];
                               // Copy other relevant properties...
                               // current_jet.jvt = this->jvt[i];
                               // current_jet.width = this->width[i];
                               // current_jet.true_flavor = this->true_flavor[i];

                               bjets.push_back(current_jet);
                         }
                 }
                 return bjets;
           }
     };

Accessing ROOT Data
-------------------

The `particle_template` base class handles the connection to ROOT TTrees. By using `add_leaf`, you map class member variables to the corresponding leaf names in the TTree:

.. code-block:: cpp

     // Inside your particle template class constructor (e.g., Jet()):
     // Map the member variable 'variable_name' to the ROOT leaf 'root_leaf_name'
     this->add_leaf("variable_name", "root_leaf_name");

     // When processing an event, access the data for the i-th particle:
     // float value = this->variable_name[i]; // For collections (like Jets)
     // float value = this->variable_name;    // For single instances (if used directly)

Registering in Event Templates
------------------------------

Particle collections must be registered within an `event_template` to be populated during event processing. Associate the collection with the TTree leaf that stores the number of particles of that type:

.. code-block:: cpp

     #include "AnalysisG/Templates/event_template.h"
     // Include headers for your particle collections
     #include "Jets.h"
     #include "Electrons.h" // Assuming Electrons.h exists
     #include "Muons.h"     // Assuming Muons.h exists

     class MyEvent : public event_template
     {
     public:
           MyEvent()
           {
                 // Instantiate particle collections
                 this->jets = new Jets();
                 this->electrons = new Electrons(); // Assuming Electrons class exists
                 this->muons = new Muons();         // Assuming Muons class exists

                 // Register each collection with the leaf storing its count
                 // Format: register_particle(collection_pointer, "count_leaf_name")
                 this->register_particle(this->jets, "n_jets");
                 this->register_particle(this->electrons, "n_electrons");
                 this->register_particle(this->muons, "n_muons");
           }

           // Pointers to the particle collections
           Jets* jets = nullptr;
           Electrons* electrons = nullptr;
           Muons* muons = nullptr;

           // Destructor to clean up allocated memory (important!)
           ~MyEvent() override
           {
                delete jets;
                delete electrons;
                delete muons;
           }
     };

Using in Graph Templates
------------------------

Particle templates (or collections) are fundamental for building graphs in `graph_template`. You can define nodes based on particles and extract their properties as node features:

.. code-block:: cpp

     #include "AnalysisG/Templates/graph_template.h"
     #include "MyEvent.h" // Include your event template definition

     class MyGraph : public graph_template
     {
           // Override the method responsible for graph construction per event
           void CompileEvent() override
           {
                 // Get the typed event object
                 MyEvent* event = this->get_event<MyEvent>();
                 if (!event) return; // Safety check

                 // --- Define Nodes ---
                 // Create graph nodes for all jets in the event's jet collection
                 // Assumes 'particles' is the std::vector<particle_template*> within Jets
                 this->define_particle_nodes(&event->jets->particles);

                 // --- Add Node Features ---
                 // Example: Add the transverse momentum (pt) as a node feature
                 // Define a lambda function to extract 'pt' from a Jet object
                 auto add_pt_feature = [](float* output_buffer, Jet* jet_particle) {
                      *output_buffer = jet_particle->pt;
                 };
                 // Register this feature extractor under the name "pt"
                 this->add_node_data_feature<float, Jet>(add_pt_feature, "pt");

                 // Add other features similarly (eta, phi, btag_score, etc.)
                 auto add_eta_feature = [](float* out, Jet* j){ *out = j->eta; };
                 this->add_node_data_feature<float, Jet>(add_eta_feature, "eta");

                 auto add_btag_feature = [](float* out, Jet* j){ *out = j->btag_score; };
                 this->add_node_data_feature<float, Jet>(add_btag_feature, "btag_score");
           }
     };


More Resources
--------------

*   **API Documentation**: See the :ref:`API Reference for Particle Templates <api_reference/particle>` for detailed class information.
*   **Examples**: Refer to the code examples provided in the `/docs/examples` directory within the AnalysisG repository.