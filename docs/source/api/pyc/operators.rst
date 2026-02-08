Operators Module (PyC)
======================

High-performance tensor operations for graph neural networks and batch processing.

Overview
--------

Located in ``pyc/pyc/operators/``, this module provides efficient tensor operations 
with C++ and CUDA backends.

Key Functions
-------------

Aggregation
~~~~~~~~~~~

.. code-block:: python

   import pyc.operators as ops
   
   # Sum aggregation
   result = ops.aggregate(values, indices, operation='sum')
   
   # Mean aggregation  
   result = ops.aggregate(values, indices, operation='mean')
   
   # Max aggregation
   result = ops.aggregate(values, indices, operation='max')

Scatter Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Scatter add
   result = ops.scatter_add(input, indices, values)
   
   # Scatter mean
   result = ops.scatter_mean(input, indices, values)

Graph Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Edge-to-node aggregation
   node_features = ops.edge_to_node(edge_features, edge_indices)
   
   # Node-to-edge broadcasting
   edge_features = ops.node_to_edge(node_features, edge_indices)

See Also
--------

* :doc:`graph` - Graph algorithms
* :doc:`physics` - Physics calculations
