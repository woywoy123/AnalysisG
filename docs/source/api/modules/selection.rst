Selection Module
===============

The Selection module implements algorithms and criteria for selecting events of interest in physics analyses.

Overview
--------

This module provides tools for defining, configuring, and applying event selection criteria to filter physical events. It enables users to implement custom selection algorithms tailored to specific physics analyses.

Key Components
-------------

selection_template class
~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: selection_template
   :members:
   :protected-members:
   :undoc-members:

Main Functionalities
-------------------

Selection Criteria Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~

The module provides methods for defining event selection criteria:

- Physics-motivated selection rules based on particle properties
- Kinematic cuts on transverse momentum, energy, and angular distributions
- Topological selections based on event structure and particle multiplicity
- Combined selection criteria with logical operations (AND, OR, NOT)

Event Filtering
~~~~~~~~~~~~

Functionality for applying selection criteria to event collections:

- ``select()``: Applies selection criteria to events
- Support for both single-event and batch processing
- Tracking of selection efficiencies and cut flow

Cut Flow Analysis
~~~~~~~~~~~~~~

Tools for analyzing the effect of sequential selection cuts:

- Recording of event counts after each selection stage
- Calculation of selection efficiencies
- Generation of cut flow tables and visualizations

Custom Selection Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~

Support for implementing custom selection algorithms:

- Framework for defining custom selection classes
- Integration with the overall analysis workflow
- Access to all event properties for complex selection logic

Usage Example
------------

.. code-block:: cpp

    #include <templates/selection_template.h>
    #include <templates/event_template.h>
    
    class top_selection : public selection_template {
    public:
        // Custom selection logic for top quark events
        bool select(event_template* event) override {
            // Count high-pT jets
            int n_jets = 0;
            for (auto& jet : event->jets) {
                if (jet.pt > 30.0 && std::abs(jet.eta) < 2.5) {
                    n_jets++;
                }
            }
            
            // Count b-tagged jets
            int n_bjets = 0;
            for (auto& jet : event->jets) {
                if (jet.pt > 30.0 && std::abs(jet.eta) < 2.5 && jet.btag > 0.8) {
                    n_bjets++;
                }
            }
            
            // Count leptons
            int n_leptons = 0;
            for (auto& lepton : event->leptons) {
                if (lepton.pt > 25.0 && std::abs(lepton.eta) < 2.4) {
                    n_leptons++;
                }
            }
            
            // Selection criteria for semileptonic top events
            return (n_jets >= 4 && n_bjets >= 1 && n_leptons == 1);
        }
    };
    
    // Using the selection in an analysis
    void apply_selection(std::vector<event_template*>* events) {
        top_selection* selection = new top_selection();
        
        std::vector<event_template*> selected_events;
        for (auto& event : *events) {
            if (selection->select(event)) {
                selected_events.push_back(event);
            }
        }
        
        std::cout << "Selected " << selected_events.size() << " out of " << events->size() << " events." << std::endl;
        
        delete selection;
    }