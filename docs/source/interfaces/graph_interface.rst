Graph Interface
===============

The Graph Interface provides functionality for creating graph-based representations of physics events.

Overview
--------

Graphs are a powerful way to represent the relationships between particles in an event. The GraphTemplate class supports:

* Node-based particle representation
* Edge connectivity between particles
* Feature extraction for nodes and edges
* Integration with Graph Neural Networks

Core GraphTemplate Class
------------------------

File Location
~~~~~~~~~~~~~

* **Cython Implementation**: ``src/AnalysisG/core/graph_template.pyx``
* **Cython Header**: ``src/AnalysisG/core/graph_template.pxd``

Class Definition
~~~~~~~~~~~~~~~~

.. class:: GraphTemplate

   Base class for graph representations of physics events.

Properties
~~~~~~~~~~

.. property:: Nodes
   
   List of graph nodes (typically particles).
   
   :type: list

.. property:: Edges
   
   List of edges connecting nodes.
   
   :type: list[tuple]

.. property:: NodeFeatures
   
   Feature matrix for nodes.
   
   :type: tensor

.. property:: EdgeFeatures
   
   Feature matrix for edges.
   
   :type: tensor

Methods to Override
~~~~~~~~~~~~~~~~~~~

.. method:: build_graph()
   
   Construct the graph structure.
   
   Override this method to define how particles are connected in the graph.

Usage Example
-------------

.. code-block:: python

   from AnalysisG.core import GraphTemplate
   
   class MyGraph(GraphTemplate):
       def build_graph(self):
           # Add all particles as nodes
           for particle in self.event.Particles:
               self.add_node(particle)
           
           # Connect particles based on ΔR
           for i, p1 in enumerate(self.Nodes):
               for j, p2 in enumerate(self.Nodes):
                   if i < j and p1.DeltaR(p2) < 0.4:
                       self.add_edge(i, j)

See Also
--------

* :doc:`../core/graph_template`: Core GraphTemplate implementation
* :doc:`../graphs/overview`: Concrete graph implementations
