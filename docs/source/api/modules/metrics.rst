Metrics Module
=============

The Metrics module implements performance evaluation and metric tracking for model evaluation and physics analysis results.

Overview
--------

This module provides tools for computing, tracking, and visualizing various metrics used to evaluate the performance of machine learning models and physics analyses. It works closely with the Plotting module to generate visual representations of performance metrics.

Key Components
-------------

metrics class
~~~~~~~~~~

.. doxygenclass:: metrics
   :members:
   :protected-members:
   :undoc-members:

metric_template class
~~~~~~~~~~~~~~~~~

.. doxygenclass:: metric_template
   :members:
   :protected-members:
   :undoc-members:

Main Functionalities
-------------------

Performance Metrics
~~~~~~~~~~~~~~~~

The module implements various performance metrics for machine learning models:

- Classification metrics (accuracy, precision, recall, F1-score)
- Regression metrics (mean squared error, mean absolute error)
- Physics-specific metrics (signal efficiency, background rejection)
- Custom metric definitions for specialized analyses

Metric Tracking
~~~~~~~~~~~~

Functionality for tracking metrics during model training and evaluation:

- Epoch-wise tracking of training and validation metrics
- K-fold cross-validation support
- Statistical aggregation of metrics across multiple runs

Metric Visualization
~~~~~~~~~~~~~~~~

Integration with the Plotting module for metric visualization:

- Learning curves over training epochs
- ROC curves for classification performance
- Precision-recall curves
- Distribution comparisons between prediction and ground truth

Custom Metrics Definition
~~~~~~~~~~~~~~~~~~~~~

Support for defining custom metrics specific to a particular physics analysis:

- Flexible API for implementing new metric calculations
- Integration with the training and evaluation workflow
- Consistent reporting format across different metric types

Usage Example
------------

.. code-block:: cpp

    #include <templates/metric_template.h>
    
    void evaluate_model_performance(model_template* model, std::vector<graph_t*>* data) {
        metric_template* metric = new metric_template();
        
        // Configure the metric
        metric->k_fold = 5;  // 5-fold cross-validation
        metric->target_mode = mode_enum::evaluation;
        
        // Register data for evaluation
        metric->register_data(data);
        
        // Compute metrics
        metric->compute(model);
        
        // Generate ROC curve
        metric->build_ROC("signal_prediction");
        
        // Get accuracy metric
        float accuracy = metric->accuracy();
        std::cout << "Model accuracy: " << accuracy << std::endl;
        
        // Get per-class metrics
        std::map<std::string, float> class_metrics = metric->class_metrics();
        
        delete metric;
    }