Model Module
===========

The Model module is responsible for defining machine learning models, training routines, and inference infrastructure for the AnalysisG framework.

Overview
--------

This module provides the foundation for implementing Graph Neural Networks and other machine learning models for high energy physics analysis. It handles model definition, training, optimization, and inference processes.

Key Components
-------------

model_template class
~~~~~~~~~~~~~~~~~

.. doxygenclass:: model_template
   :members:
   :protected-members:
   :undoc-members:

Main Functionalities
-------------------

Model Definition
~~~~~~~~~~~~~

The module provides a flexible interface for defining machine learning models:

- Support for various neural network architectures, including Graph Neural Networks
- Integration with PyTorch's neural network modules
- Custom layer definition for physics-specific operations

Training Management
~~~~~~~~~~~~~~~~

The module handles the training process for machine learning models:

- Batch processing and optimization
- Loss function computation and gradient updates
- Training metrics tracking and reporting

Forward Pass and Inference
~~~~~~~~~~~~~~~~~~~~~~~

Methods for executing model inference on input data:

- ``forward()``: Processes input data through the model
- Support for both training and evaluation modes
- Efficient batch processing for multiple inputs

Feature Registration
~~~~~~~~~~~~~~~~~

The module supports registration and tracking of various model features:

- ``prediction_graph_feature()``: Register graph-level predictions
- ``prediction_node_feature()``: Register node-level predictions
- ``prediction_edge_feature()``: Register edge-level predictions

Model Persistence
~~~~~~~~~~~~~

Functionality for saving and loading model states:

- ``save_state()``: Save model parameters and optimization state
- ``restore_state()``: Restore model from saved checkpoint

Usage Example
------------

.. code-block:: cpp

    #include <templates/model_template.h>
    
    // Create and configure a model
    model_template* create_model(model_settings_t* settings) {
        model_template* model = new model_template();
        
        // Import settings
        model->import_settings(settings);
        
        // Register model modules
        torch::nn::Sequential graph_net = torch::nn::Sequential(
            torch::nn::Linear(input_dim, hidden_dim),
            torch::nn::ReLU(),
            torch::nn::Linear(hidden_dim, output_dim)
        );
        
        model->register_module(graph_net);
        
        // Configure optimizer
        optimizer_params_t opt_params;
        opt_params.learning_rate = 0.001;
        opt_params.weight_decay = 1e-5;
        
        model->set_optimizer("Adam");
        model->initialize(&opt_params);
        
        return model;
    }