Plotting Module
===============

The modules plotting package provides advanced plotting capabilities beyond the core plotting module.

Overview
--------

Located in ``src/AnalysisG/modules/plotting/``, this module extends core plotting with:

- Multi-panel figures
- Complex plot layouts
- Advanced styling
- Publication-ready figures
- Specialized physics plots

This module builds on top of ``AnalysisG.core.plotting`` with additional functionality 
for complex visualizations.

Advanced Features
-----------------

Multi-Panel Figures
~~~~~~~~~~~~~~~~~~~

Create figures with multiple subplots:

.. code-block:: python

   from AnalysisG.modules.plotting import MultiPanel
   
   # Create 2x2 panel figure
   fig = MultiPanel(rows=2, cols=2)
   
   # Add plots to each panel
   fig.AddHistogram(hist1, row=0, col=0, title="Panel 1")
   fig.AddHistogram(hist2, row=0, col=1, title="Panel 2")
   fig.AddHistogram(hist3, row=1, col=0, title="Panel 3")
   fig.AddHistogram(hist4, row=1, col=1, title="Panel 4")
   
   fig.SaveFigure("multipanel.png")

Correlation Matrices
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.modules.plotting import CorrelationMatrix
   
   # Create correlation matrix
   corr = CorrelationMatrix()
   corr.SetVariables(["pt", "eta", "phi", "mass"])
   corr.SetData(correlation_data)
   
   corr.Draw()
   corr.SaveFigure("correlations.png")

Distribution Comparisons
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.modules.plotting import ComparisonPlot
   
   comp = ComparisonPlot()
   comp.AddDistribution(data_hist, label="Data", style="points")
   comp.AddDistribution(mc_hist, label="MC", style="fill")
   comp.AddRatioPanel()  # Add data/MC ratio
   
   comp.Draw()
   comp.SaveFigure("comparison.png")

C++ Backend
-----------

The plotting module uses C++ (``plotting.cxx``) for:

- Efficient histogram operations
- Large dataset handling
- Memory-efficient storage
- Fast rendering

Integration with Analysis
--------------------------

.. code-block:: python

   from AnalysisG import Analysis
   from AnalysisG.modules.plotting import AnalysisPlotter
   
   ana = Analysis()
   # ... configure analysis ...
   
   # Add plotting
   plotter = AnalysisPlotter()
   plotter.OutputDirectory = "plots/"
   ana.AddPlotter(plotter)

See Also
--------

* :doc:`../core/plotting` - Core plotting module
* `matplotlib documentation <https://matplotlib.org/>`_
