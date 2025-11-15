Models Module
=============

The Models module contains Graph Neural Network implementations for
high-energy physics applications.

For complete API reference, see the Doxygen-generated HTML documentation in ``doxygen-docs/html/``.

Model Implementations
---------------------

GRIFT Model
~~~~~~~~~~~

Graph Reconstruction and Inference for Tops (GRIFT).

**Location**: ``src/AnalysisG/models/grift/``

Architecture
^^^^^^^^^^^^

* **Node Encoder**: Encodes particle features
* **Message Function**: Computes edge messages
* **RNN Layers**: Recurrent processing
* **Output Layers**: Classification heads

Key Parameters
^^^^^^^^^^^^^^

* ``_hidden`` - Hidden layer dimension (default: 1024)
* ``_xrec`` - Recurrent state dimension (default: 128)
* ``_xin`` - Input feature dimension (default: 6)
* ``_xout`` - Output dimension (default: 2)
* ``_xtop`` - Top multiplicity classes (default: 5)
* ``drop_out`` - Dropout rate (default: 0.01)

Control Flags
^^^^^^^^^^^^^

* ``is_mc`` - Monte Carlo mode
* ``init`` - Initialization status
* ``pagerank`` - PageRank integration

Recursive GNN Model
~~~~~~~~~~~~~~~~~~~

Recursive Graph Neural Network architecture.

**Location**: ``src/AnalysisG/models/RecursiveGraphNeuralNetwork/``

Features
^^^^^^^^

* Multiple message-passing iterations
* Graph coarsening support
* Attention mechanisms
* Residual connections

Model Interface
---------------

All models inherit from ``model_template`` and implement:

* ``forward()`` - Forward pass computation
* ``clone()`` - Model cloning for multi-process training

Usage Example
-------------

.. code-block:: cpp

   // Create and configure model
   auto* model = new grift();
   model->_hidden = 512;
   model->drop_out = 0.1;
   
   // Forward pass
   model->forward(&graph_data);
   
   // Access predictions
   auto edge_scores = graph_data.edge_predictions;
   auto ntop_pred = graph_data.ntop_prediction;

Custom Models
-------------

To implement custom models:

1. Inherit from ``model_template``
2. Define neural network layers
3. Implement ``forward()`` method
4. Implement ``clone()`` for distributed training
5. Register layer parameters

Example:

.. code-block:: cpp

   class MyModel : public model_template {
       public:
           void forward(graph_t* graph) override {
               auto x = (*encoder)(graph->node_features);
               // ... model logic ...
           }
       
       private:
           torch::nn::Sequential* encoder = nullptr;
   };
