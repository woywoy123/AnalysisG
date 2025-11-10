Models
======

Machine learning model implementations for physics analysis.

Overview
--------

The ``models`` module contains neural network architectures:

- **RecursiveGraphNeuralNetwork**: Recursive message-passing GNN
- **grift**: Graph Inference with Feature Transformation

Model Architecture
------------------

Recursive Graph Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A recursive message-passing architecture designed for physics graphs:

- Multiple message-passing rounds
- Node and edge updates
- Graph-level readout
- Customizable aggregation

GRIFT Model
~~~~~~~~~~~

Graph Inference with Feature Transformation model for learning on physics graphs.

Model Integration
-----------------

Models in AnalysisG:

1. Inherit from ``model_template``
2. Define network architecture in C++/Python
3. Implement forward pass
4. Specify loss functions and metrics
5. Configure optimizers

Training Workflow
-----------------

The model template handles:

- **Data Loading**: Batching and shuffling
- **Training Loop**: Forward/backward passes
- **Validation**: Periodic evaluation
- **Checkpointing**: Saving best models
- **Logging**: Metrics and losses
- **Early Stopping**: Based on validation performance

Models can be trained using:

- Standard supervised learning
- Semi-supervised learning
- Self-supervised learning
- Transfer learning

Inference
---------

Trained models can be used for:

- Batch inference on new data
- Real-time event classification
- Feature extraction
- Uncertainty quantification
