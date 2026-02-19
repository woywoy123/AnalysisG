Metric Template Module (C++)
=============================

The Metric Template module provides the C++ implementation for evaluation metrics.

Overview
--------

Located in ``src/AnalysisG/modules/metric/``, this module implements metric template 
functionality in C++:

- Evaluation metric interface
- Performance tracking
- Statistical computations
- Visualization support

Purpose
-------

The metric module enables:

- Defining custom evaluation metrics
- Tracking training/validation performance
- Computing statistical measures
- Generating evaluation plots

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/metric/cxx/*.cxx`` - Metric implementations
- ``src/AnalysisG/modules/metric/include/templates/*.h`` - Metric template headers

**Python Binding**

- ``src/AnalysisG/core/metric_template.pyx`` - Cython wrapper
- ``src/AnalysisG/core/metric_template.pxd`` - Cython declarations

Key Classes
-----------

**metric_template**

Base template for metrics:

.. code-block:: cpp

   class metric_template {
   public:
       // Metric configuration
       cproperty<std::map<std::string, std::string>, metric_template> run_names;
       cproperty<std::vector<std::string>, metric_template> variables;
       cproperty<std::string, metric_template> name;
       
       // Data storage
       std::map<std::string, std::map<std::string, std::vector<double>>> data;
       
       // Virtual methods
       virtual void calculate(torch::Tensor predictions, torch::Tensor targets);
       virtual void reset();
       virtual double get_value(std::string variable, std::string run);
       
       // Serialization
       virtual void save(std::string path);
       virtual void load(std::string path);
       
       // ROOT integration
       virtual void interpret_root(std::string path);
   };

Common Metrics
--------------

**Classification Metrics**

Accuracy:

.. code-block:: cpp

   class AccuracyMetric : public metric_template {
       void calculate(torch::Tensor preds, torch::Tensor targets) override {
           auto correct = (preds.argmax(1) == targets).sum();
           double acc = correct.item<double>() / targets.size(0);
           data[current_run]["accuracy"].push_back(acc);
       }
   };

Precision/Recall/F1:

.. code-block:: cpp

   class ClassificationMetrics : public metric_template {
       void calculate(torch::Tensor preds, torch::Tensor targets) override {
           // Compute confusion matrix
           auto pred_labels = preds.argmax(1);
           
           // Calculate TP, FP, TN, FN
           auto tp = ((pred_labels == 1) & (targets == 1)).sum().item<double>();
           auto fp = ((pred_labels == 1) & (targets == 0)).sum().item<double>();
           auto tn = ((pred_labels == 0) & (targets == 0)).sum().item<double>();
           auto fn = ((pred_labels == 0) & (targets == 1)).sum().item<double>();
           
           // Precision, Recall, F1
           double precision = tp / (tp + fp);
           double recall = tp / (tp + fn);
           double f1 = 2 * (precision * recall) / (precision + recall);
           
           data[current_run]["precision"].push_back(precision);
           data[current_run]["recall"].push_back(recall);
           data[current_run]["f1"].push_back(f1);
       }
   };

**Regression Metrics**

Mean Squared Error:

.. code-block:: cpp

   class MSEMetric : public metric_template {
       void calculate(torch::Tensor preds, torch::Tensor targets) override {
           auto mse = torch::mse_loss(preds, targets);
           data[current_run]["mse"].push_back(mse.item<double>());
       }
   };

Mean Absolute Error:

.. code-block:: cpp

   class MAEMetric : public metric_template {
       void calculate(torch::Tensor preds, torch::Tensor targets) override {
           auto mae = torch::mean(torch::abs(preds - targets));
           data[current_run]["mae"].push_back(mae.item<double>());
       }
   };

Usage Example
-------------

**Defining a Metric**

.. code-block:: cpp

   #include <templates/metric_template.h>
   
   class CustomMetric : public metric_template {
   public:
       CustomMetric() {
           name = "CustomMetric";
           run_names = {{"train", "Training"}, {"valid", "Validation"}};
           variables = {"loss", "accuracy", "auc"};
       }
       
       void calculate(torch::Tensor preds, torch::Tensor targets) override {
           // Compute metrics
           double loss = compute_loss(preds, targets);
           double acc = compute_accuracy(preds, targets);
           double auc = compute_auc(preds, targets);
           
           // Store values
           data[current_run]["loss"].push_back(loss);
           data[current_run]["accuracy"].push_back(acc);
           data[current_run]["auc"].push_back(auc);
       }
   };

**Using the Metric**

.. code-block:: cpp

   // Create metric
   CustomMetric metric;
   
   // Training loop
   for (int epoch = 0; epoch < num_epochs; ++epoch) {
       // Training
       metric.current_run = "train";
       for (auto& batch : train_loader) {
           auto output = model(batch.x);
           metric.calculate(output, batch.y);
       }
       
       // Validation
       metric.current_run = "valid";
       for (auto& batch : valid_loader) {
           auto output = model(batch.x);
           metric.calculate(output, batch.y);
       }
   }
   
   // Get results
   double train_acc = metric.get_value("accuracy", "train");
   double valid_acc = metric.get_value("accuracy", "valid");
   
   // Save metrics
   metric.save("metrics.json");

Data Storage
------------

Metrics are stored hierarchically:

.. code-block:: cpp

   std::map<std::string, std::map<std::string, std::vector<double>>> data;
   // data[run_name][variable_name] = [value1, value2, ...]

Example structure:

.. code-block:: json

   {
       "train": {
           "loss": [0.5, 0.4, 0.3, 0.2],
           "accuracy": [0.7, 0.75, 0.8, 0.85]
       },
       "valid": {
           "loss": [0.6, 0.5, 0.45, 0.4],
           "accuracy": [0.65, 0.7, 0.72, 0.75]
       }
   }

ROOT Integration
----------------

Export metrics to ROOT files:

.. code-block:: cpp

   // Save to ROOT
   metric.interpret_root("metrics.root");

This creates:

- TGraphs for each variable
- TH1F histograms
- TNtuple for data
- Automatic styling

Visualization
-------------

Metrics can be plotted:

.. code-block:: cpp

   // Plot training curves
   metric.plot("accuracy");
   metric.plot_all();

**Multi-Run Comparison**

.. code-block:: cpp

   // Compare multiple runs
   metric.compare_runs({"train", "valid"}, "accuracy");

Serialization
-------------

**Save Metrics**

.. code-block:: cpp

   // JSON format
   metric.save("metrics.json");
   
   // Pickle format (Python)
   metric.save("metrics.pkl");
   
   // ROOT format
   metric.interpret_root("metrics.root");

**Load Metrics**

.. code-block:: cpp

   // Load from file
   metric.load("metrics.json");

Statistical Analysis
--------------------

Compute statistics:

.. code-block:: cpp

   // Mean
   double mean = metric.compute_mean("accuracy", "train");
   
   // Standard deviation
   double std = metric.compute_std("accuracy", "train");
   
   // Min/Max
   double min_val = metric.compute_min("accuracy", "train");
   double max_val = metric.compute_max("accuracy", "train");
   
   // Moving average
   auto smoothed = metric.moving_average("loss", "train", window=5);

Integration with Python
-----------------------

The C++ metric_template is wrapped in Python:

.. code-block:: python

   from AnalysisG.core.metric_template import MetricTemplate
   
   class MyMetric(MetricTemplate):
       def __init__(self):
           super().__init__()
           self.RunNames = {"train": "Training", "valid": "Validation"}
           self.Variables = ["loss", "accuracy"]

Epoch Tracking
--------------

Track metrics per epoch:

.. code-block:: cpp

   class EpochMetric : public metric_template {
       int current_epoch = 0;
       
       void on_epoch_end() {
           // Compute epoch averages
           double avg_loss = compute_average("loss", current_run);
           epoch_data[current_epoch][current_run]["avg_loss"] = avg_loss;
           current_epoch++;
       }
   };

Early Stopping
--------------

Implement early stopping based on metrics:

.. code-block:: cpp

   class EarlyStoppingMetric : public metric_template {
       int patience = 10;
       int wait = 0;
       double best_loss = INFINITY;
       
       bool should_stop() {
           double current_loss = get_value("loss", "valid");
           
           if (current_loss < best_loss) {
               best_loss = current_loss;
               wait = 0;
               return false;
           }
           
           wait++;
           return wait >= patience;
       }
   };

See Also
--------

* :doc:`../core/metric_template` - Python MetricTemplate wrapper
* :doc:`model` - Model template
* :doc:`roc` - ROC curve calculations
* :doc:`../modules/metrics` - Specific metric implementations
