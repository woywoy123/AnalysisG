ROC Module (C++)
=================

The ROC module provides Receiver Operating Characteristic curve calculations.

Overview
--------

Located in ``src/AnalysisG/modules/roc/``, this module implements ROC curve functionality 
in C++:

- ROC curve computation
- AUC (Area Under Curve) calculation
- Optimal threshold selection
- Multi-class ROC curves

Purpose
-------

The ROC module enables:

- Binary and multi-class classification evaluation
- Signal vs background discrimination
- Threshold optimization
- Performance visualization

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/roc/cxx/*.cxx`` - ROC implementations
- ``src/AnalysisG/modules/roc/include/plotting/*.h`` - ROC headers

**Python Binding**

- ``src/AnalysisG/core/roc.pyx`` - Cython wrapper
- ``src/AnalysisG/core/roc.pxd`` - Cython declarations

Key Classes
-----------

**roc_curve**

ROC curve calculator:

.. code-block:: cpp

   class roc_curve {
   public:
       // ROC computation
       void compute(std::vector<double> scores, 
                   std::vector<int> labels);
       
       // Results
       std::vector<double> fpr;   // False positive rate
       std::vector<double> tpr;   // True positive rate
       std::vector<double> thresholds;
       
       // Metrics
       double auc();              // Area under curve
       double best_threshold();   // Optimal threshold
       
       // Visualization
       void plot(std::string filename);
   };

Usage Example
-------------

**Basic ROC Curve**

.. code-block:: cpp

   #include <plotting/roc.h>
   
   // Compute ROC curve
   roc_curve roc;
   roc.compute(predicted_scores, true_labels);
   
   // Get AUC
   double auc_value = roc.auc();
   std::cout << "AUC: " << auc_value << std::endl;
   
   // Find best threshold
   double threshold = roc.best_threshold();
   std::cout << "Best threshold: " << threshold << std::endl;
   
   // Plot
   roc.plot("roc_curve.png");

ROC Metrics
-----------

**Area Under Curve (AUC)**

Compute the area under the ROC curve:

.. code-block:: cpp

   double auc_value = roc.auc();
   // Range: [0, 1], where 1.0 is perfect classification

**True Positive Rate (TPR)**

.. math::

   TPR = \frac{TP}{TP + FN}

Also known as Sensitivity or Recall.

**False Positive Rate (FPR)**

.. math::

   FPR = \frac{FP}{FP + TN}

**Optimal Threshold**

Find threshold that maximizes Youden's J statistic:

.. math::

   J = TPR - FPR

.. code-block:: cpp

   double optimal = roc.best_threshold();

Multi-Class ROC
---------------

For multi-class problems:

**One-vs-Rest**

.. code-block:: cpp

   std::map<int, roc_curve> roc_curves;
   
   for (int class_id = 0; class_id < num_classes; ++class_id) {
       // Binary labels: class_id vs rest
       auto binary_labels = make_binary(true_labels, class_id);
       
       roc_curves[class_id].compute(scores[class_id], binary_labels);
   }
   
   // Average AUC
   double avg_auc = 0;
   for (auto& [id, roc] : roc_curves) {
       avg_auc += roc.auc();
   }
   avg_auc /= num_classes;

**One-vs-One**

.. code-block:: cpp

   for (int i = 0; i < num_classes; ++i) {
       for (int j = i + 1; j < num_classes; ++j) {
           // Filter to only classes i and j
           auto filtered_scores = filter_classes(scores, {i, j});
           auto filtered_labels = filter_classes(labels, {i, j});
           
           roc_curve roc;
           roc.compute(filtered_scores, filtered_labels);
       }
   }

Operating Points
----------------

Extract specific operating points:

.. code-block:: cpp

   // High signal efficiency
   double threshold_90 = roc.threshold_at_tpr(0.90);
   
   // Low false positive rate
   double threshold_01 = roc.threshold_at_fpr(0.01);
   
   // Balanced
   double threshold_bal = roc.threshold_at_equal_error_rate();

Significance Calculation
------------------------

For particle physics applications:

.. code-block:: cpp

   double significance(double signal, double background) {
       return signal / sqrt(signal + background);
   }
   
   // Find threshold maximizing significance
   double best_threshold = 0;
   double best_sig = 0;
   
   for (double threshold : roc.thresholds) {
       double sig = count_signal_above(scores, labels, threshold);
       double bkg = count_background_above(scores, labels, threshold);
       double s = significance(sig, bkg);
       
       if (s > best_sig) {
           best_sig = s;
           best_threshold = threshold;
       }
   }

Plotting Options
----------------

**Basic Plot**

.. code-block:: cpp

   roc.plot("roc.png");

**Customized Plot**

.. code-block:: cpp

   roc.set_title("Signal vs Background");
   roc.set_xlabel("Background Efficiency");
   roc.set_ylabel("Signal Efficiency");
   roc.set_color("blue");
   roc.set_linewidth(2);
   roc.add_diagonal();  // Add random classifier line
   roc.plot("roc_custom.png");

**Multiple ROC Curves**

.. code-block:: cpp

   // Compare multiple models
   std::vector<roc_curve> rocs;
   std::vector<std::string> labels = {"Model 1", "Model 2", "Model 3"};
   
   for (auto& model : models) {
       roc_curve roc;
       roc.compute(model.predict(X_test), y_test);
       rocs.push_back(roc);
   }
   
   plot_multiple_roc(rocs, labels, "comparison.png");

Confusion Matrix
----------------

Extract confusion matrix at threshold:

.. code-block:: cpp

   struct ConfusionMatrix {
       int tp, fp, tn, fn;
       
       double precision() { return (double)tp / (tp + fp); }
       double recall() { return (double)tp / (tp + fn); }
       double f1() { 
           double p = precision();
           double r = recall();
           return 2 * p * r / (p + r);
       }
   };
   
   ConfusionMatrix cm = roc.confusion_matrix(threshold);

Partial AUC
-----------

Compute AUC in specific FPR range:

.. code-block:: cpp

   // AUC for FPR in [0, 0.1]
   double partial_auc = roc.partial_auc(0.0, 0.1);
   
   // Useful for low false positive rate regions
   double low_fpr_auc = roc.partial_auc(0.0, 0.01);

Cross-Validation
----------------

ROC with cross-validation:

.. code-block:: cpp

   std::vector<roc_curve> cv_rocs;
   
   for (auto& [train, test] : cv_splits) {
       model.train(train);
       auto scores = model.predict(test.X);
       
       roc_curve roc;
       roc.compute(scores, test.y);
       cv_rocs.push_back(roc);
   }
   
   // Average ROC
   auto avg_roc = average_roc_curves(cv_rocs);
   double mean_auc = avg_roc.auc();
   double std_auc = compute_auc_std(cv_rocs);

Integration with Python
-----------------------

The C++ ROC module is wrapped in Python:

.. code-block:: python

   from AnalysisG.core.roc import ROC
   
   roc = ROC()
   roc.compute(scores, labels)
   print(f"AUC: {roc.auc()}")
   roc.plot("roc.png")

See Also
--------

* :doc:`../core/roc` - Python ROC wrapper
* :doc:`metric` - Metric template
* :doc:`../modules/metrics` - Other metrics
