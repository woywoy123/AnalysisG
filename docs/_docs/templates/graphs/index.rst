Physics Graphs
=============

The graph system in AnalysisG provides representations of physics events as mathematical graphs, which are particularly suitable for machine learning applications. The framework includes several graph templates found in the ``src/AnalysisG/graphs/`` directory.

.. toctree::
   :maxdepth: 1
   
   ssml_mc20/index
   templates/index

Graph Model
----------

Graphs in AnalysisG follow the ``graph_template`` base class, which defines a standard interface. Each specific graph implementation inherits from this template and implements the required methods.

.. code-block:: cpp

   // Base graph template interface
   class graph_template {
      public:
         virtual graph_template* clone() = 0;
         virtual void CompileEvent() = 0;
         // ...
   };

Graph Structure
--------------

A physics graph in AnalysisG consists of:

* **Nodes**: Representing physics objects (particles, jets, etc.)
* **Edges**: Connections between physics objects
* **Features**: Properties associated with nodes and edges
* **Global attributes**: Properties of the entire graph

Key Graph Types
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Graph Type
     - Description
   * - ``graph_jets``
     - Standard jet-based graph representation
   * - ``graph_jets_nonu``
     - Jet-based graph without neutrinos
   * - ``graph_jets_detector_lep``
     - Detector-level jets with leptons
   * - ``graph_detector``
     - Detector-level objects graph

Creating Custom Graphs
---------------------

To create a custom graph type:

1. Inherit from ``graph_template``
2. Implement the required virtual methods
3. Define node and edge features
4. Implement the ``CompileEvent`` method to build graph structure

Graph Features
-------------

Graph features are defined separately from the graph structure:

* **Node features** (e.g., pt, eta, phi, energy, particle type)
* **Edge features** (e.g., distance, invariant mass, relative angles)
* **Graph features** (e.g., total energy, multiplicity)

Example Graph Template
--------------------

.. code-block:: cpp

   // Basic graph template implementation
   class graph_jets: public graph_template
   {
       public:
           graph_jets() { this->name = "graph_jets"; }
           ~graph_jets() {}
           graph_template* clone() { return (graph_template*)new graph_jets(); }
           void CompileEvent() {
               // Implement graph building logic
           }
   };