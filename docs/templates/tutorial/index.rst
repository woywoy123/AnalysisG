.. _tutorials/index:

====================
Tutorials
====================

These tutorials show common AnalysisG use cases with examples.

Basics
------

.. toctree::
    :maxdepth: 1

    basics/installation
    basics/first_analysis
    basics/working_with_root

Events and Particles
--------------------

.. toctree::
    :maxdepth: 1

    events/defining_events
    events/particle_collections
    events/event_selections

Graphs and GNNs
---------------

.. toctree::
    :maxdepth: 1

    graphs/creating_graphs
    graphs/defining_features
    graphs/custom_topology

Machine Learning
----------------

.. toctree::
    :maxdepth: 1

    ml/training_gnns
    ml/evaluating_models
    ml/hyperparameter_tuning

Example Scripts
---------------

The `/workspaces/AnalysisG/docs/examples` directory has example scripts showing different AnalysisG features:

* **test_particles.py**: Shows basic usage of `ParticleTemplate`
* **test_events.py**: Shows how to define and process events
* **test_graphs.py**: Explains creating and manipulating graphs
* **test_training.py**: A complete example of training a GNN model

You can run these scripts directly to see how the framework works:

.. code-block:: bash

    python docs/examples/test_particles.py

