Metrics Module
==============

The metrics module provides evaluation metrics for analysis and machine learning.

Overview
--------

Located in ``src/AnalysisG/modules/metrics/``, this module implements:

- Mass reconstruction plots
- Loss function visualizations
- Custom metric calculations
- Performance evaluation tools

The module integrates with both the Analysis framework and standalone usage.

C++ Implementation
------------------

The module includes C++ implementations in:

- ``metrics.cxx``: Core metric calculations
- ``mass_plots.cxx``: Mass distribution plotting
- ``loss_plots.cxx``: Loss visualization

API Reference
-------------

Mass Plots
~~~~~~~~~~

Functionality for plotting reconstructed masses:

.. code-block:: python

   from AnalysisG.modules import metrics
   
   # Create mass plot
   mass_plot = metrics.MassPlot()
   mass_plot.Title = "Top Quark Mass"
   mass_plot.xTitle = "Mass [GeV]"
   
   # Fill with reconstructed masses
   for event in events:
       for top_candidate in event.top_candidates:
           mass_plot.Fill(top_candidate.mass)

Loss Plots
~~~~~~~~~~

Visualization of training losses:

.. code-block:: python

   from AnalysisG.modules import metrics
   
   # Create loss plot
   loss_plot = metrics.LossPlot()
   
   # Add training losses
   for epoch, loss in enumerate(training_losses):
       loss_plot.AddTrainingLoss(epoch, loss)
   
   # Add validation losses
   for epoch, loss in enumerate(validation_losses):
       loss_plot.AddValidationLoss(epoch, loss)
   
   # Draw
   loss_plot.Draw()
   loss_plot.SaveFigure("training_losses.png")

Custom Metrics
~~~~~~~~~~~~~~

Implement custom metrics using MetricTemplate:

.. code-block:: python

   from AnalysisG.core import MetricTemplate
   
   class MyMetrics(MetricTemplate):
       def __init__(self):
           super().__init__()
       
       def Calculate(self, predictions, targets):
           # Custom metric calculations
           accuracy = calculate_accuracy(predictions, targets)
           f1_score = calculate_f1(predictions, targets)
           
           return {
               'accuracy': accuracy,
               'f1_score': f1_score
           }

Integration Example
-------------------

.. code-block:: python

   from AnalysisG.core import Analysis, MetricTemplate
   
   class CustomMetrics(MetricTemplate):
       def Calculate(self, preds, targets):
           return {'custom_metric': compute_metric(preds, targets)}
   
   # Add to analysis
   ana = Analysis()
   metrics = CustomMetrics()
   ana.AddMetric(metrics, model)

See Also
--------

* :doc:`../core/templates` - MetricTemplate documentation
* :doc:`../core/plotting` - Plotting utilities
