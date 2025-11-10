Events
======

Event implementations for different physics analyses and data formats.

Overview
--------

The ``events`` module contains concrete implementations of the event template for various physics analyses:

- **bsm_4tops**: Four-top quark BSM analysis events
- **exp_mc20**: Experimental MC20 dataset events  
- **gnn**: Graph neural network inference events
- **ssml_mc20**: Semi-supervised machine learning MC20 events

Each event implementation defines:

- Particle collections (Jets, Electrons, Muons, etc.)
- Event-level variables (MET, weights, etc.)
- Truth-level information for MC
- Build and compilation logic

BSM 4-Tops Event
----------------

Event class for four-top quark BSM physics analysis. Contains:

- Top quark collections
- Children particles from top decays
- Truth jets
- Detector-level jets, electrons, muons
- Event metadata (event number, pileup, MET)

Experimental MC20 Event
-----------------------

General-purpose event class for MC20 simulation samples. Provides:

- Standard detector objects
- Truth-level particles
- Event weights and identifiers

GNN Event
---------

Specialized event class optimized for graph neural network applications:

- Simplified particle collections
- Graph-friendly data structures
- Inference-optimized layout

SSML MC20 Event
---------------

Event class designed for semi-supervised machine learning on MC20:

- Additional particle types (leptons, jets)
- Extended truth information
- MET and kinematics

Event Relationships
-------------------

All event classes inherit from ``event_template`` and implement:

1. **clone()**: Create a copy of the event instance
2. **build()**: Populate event from ROOT data
3. **CompileEvent()**: Finalize event after building

Event classes manage particle collections through:

- Public vectors of ``particle_template*`` for user access
- Private maps for efficient particle lookup during building
- Automatic vectorization and sorting
