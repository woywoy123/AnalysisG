Graphs
======

Graph implementations for neural network training and inference.

Overview
--------

The ``graphs`` module contains concrete graph implementations for different analyses:

- **bsm_4tops**: Graph representation for four-top BSM analysis
- **exp_mc20**: Graph for experimental MC20 data
- **ssml_mc20**: Graph for semi-supervised learning on MC20

Graph Structure
---------------

Each graph implementation defines three levels of features:

1. **Node Features**: Derived from individual particles
2. **Edge Features**: Derived from particle pairs
3. **Graph Features**: Derived from the entire event

BSM 4-Tops Graph
----------------

Graph structure optimized for four-top quark final states.

Experimental MC20 Graph
-----------------------

Graph structure for general-purpose MC20 analysis.

SSML MC20 Graph
---------------

Graph structure designed for semi-supervised machine learning tasks.

Feature Engineering
-------------------

Graph classes implement feature extraction:

- **NodeFeature()**: Computes features for each node
- **EdgeFeature()**: Computes features for each edge  
- **GraphFeature()**: Computes global event features

These methods are called automatically during graph construction to populate feature tensors for neural network input.

Relationship to Events
----------------------

Graphs are built from Events:

1. Event is created and compiled from ROOT data
2. Graph template is instantiated with the Event
3. Graph extracts particles as nodes
4. Graph computes edges between nodes
5. Feature methods populate tensors

This separation allows:

- Reusing event definitions across multiple graph types
- Testing different graph topologies
- Experimenting with feature engineering
- Training multiple models on the same events
