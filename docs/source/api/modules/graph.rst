Graph Module (C++)
==================

The Graph module provides the C++ implementation of graph templates for GNN construction.

Overview
--------

Located in ``src/AnalysisG/modules/graph/``, this module implements graph template 
functionality in C++ for constructing graph structures for Graph Neural Networks:

- Graph construction from events
- Node and edge feature extraction
- Graph-level feature aggregation
- Integration with PyTorch Geometric data structures

C++ Class: graph_template
--------------------------

Header Location
~~~~~~~~~~~~~~~

``src/AnalysisG/modules/graph/include/templates/graph_template.h``

Key Properties
~~~~~~~~~~~~~~

The ``graph_template`` class uses ``cproperty`` template accessors:

.. code-block:: cpp

   cproperty<std::string, graph_template> name;         // Graph name
   cproperty<std::string, graph_template> hash;         // Unique hash
   cproperty<long, graph_template> index;               // Graph index
   cproperty<std::string, graph_template> tree;         // Associated tree name
   cproperty<bool, graph_template> preselection;        // Pre-selection flag

Key Methods
~~~~~~~~~~~

**Template Management**

.. cpp:function:: virtual graph_template* clone()

   Create a copy of the graph template.

.. cpp:function:: virtual void build(event_template* ev)

   Build graph structure from event.

   :param ev: Event template containing particles and event data

.. cpp:function:: virtual void CompileGraph()

   Compile and finalize graph structure, preparing for ML framework.

**Graph Construction**

Graph templates define how to construct graphs from events. Subclasses override
virtual methods to specify:

- Which particles become nodes
- How edges are defined (connectivity)
- What features are extracted for nodes
- What features are computed for edges  
- What global graph-level features are included

**Feature Extraction**

The graph template coordinates feature extraction for:

- **Node features**: Per-particle properties (e.g., pt, eta, phi, energy)
- **Edge features**: Relationships between particles (e.g., ΔR, angular separation)
- **Graph features**: Event-level properties (e.g., MET, HT, jet multiplicity)

Member Variables
~~~~~~~~~~~~~~~~

.. cpp:member:: graph_t data

   Graph data structure containing nodes, edges, and features.

.. cpp:member:: std::vector<particle_template*> nodes

   Vector of particles serving as graph nodes.

.. cpp:member:: std::vector<std::pair<int, int>> edges

   Vector of edge connections (source, target) indices.

.. cpp:member:: std::map<std::string, std::vector<double>> node_features

   Map of node feature names to feature vectors.

.. cpp:member:: std::map<std::string, std::vector<double>> edge_features

   Map of edge feature names to feature vectors.

.. cpp:member:: std::map<std::string, double> graph_features

   Map of graph-level feature names to values.

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/graph/cxx/graph_template.cxx`` - Main implementation
- ``src/AnalysisG/modules/graph/cxx/properties.cxx`` - Property implementations
- ``src/AnalysisG/modules/graph/cxx/features.cxx`` - Feature extraction

**Python Binding**

- ``src/AnalysisG/core/graph_template.pyx`` - Cython wrapper
- ``src/AnalysisG/core/graph_template.pxd`` - Cython declarations

Usage from C++
--------------

.. code-block:: cpp

   #include <templates/graph_template.h>
   #include <templates/event_template.h>
   
   // Create graph template
   graph_template* graph = new graph_template();
   graph->name = "ParticleGraph";
   graph->preselection = true;
   
   // Build graph from event
   event_template* event = /* ... */;
   graph->build(event);
   
   // Access graph structure
   for (auto& node : graph->nodes) {
       double pt = node->pt;
       double eta = node->eta;
   }
   
   for (auto& edge : graph->edges) {
       int source = edge.first;
       int target = edge.second;
   }
   
   // Compile for ML framework
   graph->CompileGraph();

Integration with Python
-----------------------

The C++ graph_template is wrapped in Python as ``GraphTemplate``:

.. code-block:: python

   from AnalysisG.core.graph_template import GraphTemplate
   
   class ParticleGraph(GraphTemplate):
       def __init__(self):
           super().__init__()
           self.PreSelection = True

Graph Construction Patterns
----------------------------

**Complete Graph**

Connect all particles to all other particles:

.. code-block:: cpp

   // In build() method
   for (size_t i = 0; i < nodes.size(); ++i) {
       for (size_t j = 0; j < nodes.size(); ++j) {
           if (i != j) {
               edges.push_back({i, j});
           }
       }
   }

**k-Nearest Neighbors**

Connect each particle to its k nearest neighbors in η-φ space:

.. code-block:: cpp

   // Calculate distances and find k nearest
   for (size_t i = 0; i < nodes.size(); ++i) {
       std::vector<std::pair<double, size_t>> distances;
       for (size_t j = 0; j < nodes.size(); ++j) {
           if (i != j) {
               double deta = nodes[i]->eta - nodes[j]->eta;
               double dphi = nodes[i]->phi - nodes[j]->phi;
               double dr = sqrt(deta*deta + dphi*dphi);
               distances.push_back({dr, j});
           }
       }
       std::sort(distances.begin(), distances.end());
       for (size_t k = 0; k < std::min(K, distances.size()); ++k) {
           edges.push_back({i, distances[k].second});
       }
   }

**Radius Graph**

Connect particles within a radius threshold:

.. code-block:: cpp

   const double R_MAX = 0.4;
   for (size_t i = 0; i < nodes.size(); ++i) {
       for (size_t j = i + 1; j < nodes.size(); ++j) {
           double deta = nodes[i]->eta - nodes[j]->eta;
           double dphi = nodes[i]->phi - nodes[j]->phi;
           double dr = sqrt(deta*deta + dphi*dphi);
           if (dr < R_MAX) {
               edges.push_back({i, j});
               edges.push_back({j, i});  // Undirected
           }
       }
   }

Feature Extraction
------------------

**Node Features**

Typical node features for particles:

- Kinematics: pt, eta, phi, energy, mass
- Particle ID: pdgid, charge
- Isolation: track isolation, calorimeter isolation
- b-tagging: b-tag score (for jets)

**Edge Features**

Typical edge features between particles:

- ΔR: Angular separation in η-φ space
- Δη, Δφ: Separation in pseudorapidity and azimuth
- Invariant mass: Combined mass of two particles
- Angular separation: cos(θ) between momenta

**Graph Features**

Typical graph-level features:

- MET: Missing transverse energy
- HT: Scalar sum of jet pt
- Multiplicities: Number of jets, leptons, b-jets
- Event weights: MC weights, pile-up weights

PyTorch Geometric Integration
------------------------------

The compiled graph can be converted to PyTorch Geometric ``Data`` objects:

.. code-block:: python

   from torch_geometric.data import Data
   import torch
   
   # After graph compilation
   x = torch.tensor(graph.node_features, dtype=torch.float)
   edge_index = torch.tensor(graph.edges, dtype=torch.long).t()
   edge_attr = torch.tensor(graph.edge_features, dtype=torch.float)
   y = torch.tensor(graph.labels, dtype=torch.long)
   
   data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

See Also
--------

* :doc:`../core/templates` - Python GraphTemplate documentation
* :doc:`event` - Event template C++ implementation
* :doc:`particle` - Particle template C++ implementation
* :doc:`../pyc/graph` - High-performance graph algorithms
