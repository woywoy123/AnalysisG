Graphs Module
=============

The Graphs module transforms physics events into graph representations suitable
for Graph Neural Network processing.

For complete API reference, see the Doxygen-generated HTML documentation in ``doxygen-docs/html/``.

Graph Implementations
---------------------

BSM Four-Top Graphs
~~~~~~~~~~~~~~~~~~~

Graph construction for BSM four-top analysis.

**Location**: ``src/AnalysisG/graphs/bsm_4tops/``

Experimental MC20 Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~

Graph construction for MC20 experimental data.

**Location**: ``src/AnalysisG/graphs/exp_mc20/``

SSML MC20 Graphs
~~~~~~~~~~~~~~~~

Graph construction for Same-Sign Multi-Lepton analysis.

**Location**: ``src/AnalysisG/graphs/ssml_mc20/``

Graph Structure
---------------

Graph objects include:

* **Nodes**: Represent particles or physics objects
* **Node Features**: Kinematic and identification properties
* **Edges**: Connections between related particles
* **Edge Features**: Relationship properties
* **Global Features**: Event-level properties

Common Features
---------------

Node Features
~~~~~~~~~~~~~

* Momentum components (px, py, pz)
* Energy
* Mass
* Particle type/flavor
* Isolation variables
* B-tagging scores (for jets)

Edge Features
~~~~~~~~~~~~~

* Angular separation (ΔR)
* Invariant mass
* Transverse momentum sum
* Relative orientation

Usage Example
-------------

.. code-block:: cpp

   // Create graph from event
   auto* graph = new GraphType();
   graph->build(event);
   
   // Access graph components
   auto nodes = graph->GetNodes();
   auto edges = graph->GetEdges();
   auto features = graph->GetFeatures();
