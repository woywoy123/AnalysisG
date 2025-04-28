Particle Module
==============

The Particle module defines particle physics objects and their properties for use in event analysis.

Overview
--------

This module provides a comprehensive set of classes and utilities for representing fundamental particles and composite objects in high energy physics. It implements standard particle properties and behaviors relevant for physics analyses.

Key Components
-------------

particle_template class
~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: particle_template
   :members:
   :protected-members:
   :undoc-members:

particle_id enum
~~~~~~~~~~~~~

.. doxygenenum:: particle_id
   :undoc-members:

Main Functionalities
-------------------

Particle Representation
~~~~~~~~~~~~~~~~~~~~

The module provides a standard representation for physics particles:

- Four-momentum representation (px, py, pz, E)
- Derived kinematic properties (pt, eta, phi, mass)
- Particle identification and classification
- Charge, spin, and other quantum numbers

Composite Object Handling
~~~~~~~~~~~~~~~~~~~~~

Support for composite particle objects:

- Reconstruction of parent particles from decay products
- Jet constituents and substructure
- Resonance reconstruction

Kinematic Calculations
~~~~~~~~~~~~~~~~~~~

Methods for common kinematic calculations:

- ``deltaR()``: Calculate angular separation between particles
- ``invariant_mass()``: Calculate invariant mass of particle systems
- Boost and rotation of particle momenta
- Transverse and longitudinal projections

Physics ID and Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for particle identification:

- PDG ID mapping and manipulation
- Lepton/hadron/jet classification
- Truth-level to detector-level particle matching

Usage Example
------------

.. code-block:: cpp

    #include <templates/particle_template.h>
    
    void analyze_particles(std::vector<particle_template*> particles) {
        // Filter for high-pT electrons
        std::vector<particle_template*> electrons;
        for (auto& particle : particles) {
            if (particle->id == particle_id::electron && particle->pt > 25.0) {
                electrons.push_back(particle);
            }
        }
        
        // Calculate invariant mass of electron pairs
        if (electrons.size() >= 2) {
            for (size_t i = 0; i < electrons.size() - 1; i++) {
                for (size_t j = i + 1; j < electrons.size(); j++) {
                    double mass = electrons[i]->invariant_mass(*electrons[j]);
                    
                    // Check if compatible with Z boson
                    if (std::abs(mass - 91.2) < 10.0) {
                        std::cout << "Z candidate found with mass: " << mass << " GeV" << std::endl;
                    }
                }
            }
        }
    }