Plotting Module
===============

The plotting module provides a high-level interface for creating physics plots using 
boost-histogram and mplhep.

Overview
--------

The ``plotting`` module (``src/AnalysisG/core/plotting.pyx``) offers:

- Object-oriented plotting interface
- Integration with boost-histogram
- ATLAS/CMS style presets via mplhep
- Automatic histogram binning
- Support for 1D and 2D histograms
- Statistical error handling
- Easy figure export

Basic Usage
-----------

Creating Histograms
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.plotting import Histogram
   
   # Create 1D histogram
   hist = Histogram()
   hist.Title = "Jet $p_T$"
   hist.xTitle = "$p_T$ [GeV]"
   hist.yTitle = "Events"
   hist.xBins = 50
   hist.xMin = 0
   hist.xMax = 500
   
   # Fill histogram
   for event in events:
       for jet in event.jets:
           hist.Fill(jet.pt)
   
   # Draw histogram
   hist.Draw()
   hist.SaveFigure("jet_pt.png")

2D Histograms
~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.plotting import Histogram2D
   
   # Create 2D histogram
   hist2d = Histogram2D()
   hist2d.Title = "Jet $\eta$ vs $\phi$"
   hist2d.xTitle = "$\eta$"
   hist2d.yTitle = "$\phi$"
   hist2d.xBins = 50
   hist2d.yBins = 50
   
   # Fill histogram
   for event in events:
       for jet in event.jets:
           hist2d.Fill(jet.eta, jet.phi)
   
   # Draw as heatmap
   hist2d.Draw()
   hist2d.SaveFigure("jet_eta_phi.png")

API Reference
-------------

Histogram Class
~~~~~~~~~~~~~~~

.. class:: Histogram

   1D histogram class.

   **Properties**

   .. attribute:: Title
      :type: str

      Histogram title.

   .. attribute:: xTitle
      :type: str

      X-axis title with LaTeX support.

   .. attribute:: yTitle
      :type: str

      Y-axis title with LaTeX support.

   .. attribute:: xBins
      :type: int

      Number of bins on x-axis.

   .. attribute:: xMin
      :type: float

      Minimum value on x-axis.

   .. attribute:: xMax
      :type: float

      Maximum value on x-axis.

   .. attribute:: Color
      :type: str

      Histogram color (matplotlib color string).

   .. attribute:: LineWidth
      :type: float

      Line width for histogram outline.

   .. attribute:: Alpha
      :type: float

      Transparency (0-1).

   .. attribute:: Style
      :type: str

      Histogram style: "step", "fill", "errorbar".

   **Methods**

   .. method:: Fill(x: float, weight: float = 1.0)

      Fill histogram with a value.

      :param x: Value to fill
      :param weight: Event weight
      :type x: float
      :type weight: float

   .. method:: Draw()

      Draw the histogram.

      Creates or updates the current figure with the histogram.

   .. method:: SaveFigure(filename: str, dpi: int = 300)

      Save figure to file.

      :param filename: Output filename
      :param dpi: Resolution in dots per inch
      :type filename: str
      :type dpi: int

      Supported formats: PNG, PDF, SVG, EPS

   .. method:: Clear()

      Clear histogram contents.

   .. method:: Normalize(norm: float = 1.0)

      Normalize histogram to given value.

      :param norm: Normalization value
      :type norm: float

   .. method:: GetIntegral() -> float

      Get integral of histogram.

      :return: Sum of all bin contents
      :rtype: float

Histogram2D Class
~~~~~~~~~~~~~~~~~

.. class:: Histogram2D

   2D histogram class.

   **Properties**

   .. attribute:: Title
      :type: str

      Histogram title.

   .. attribute:: xTitle
      :type: str

      X-axis title.

   .. attribute:: yTitle
      :type: str

      Y-axis title.

   .. attribute:: xBins
      :type: int

      Number of bins on x-axis.

   .. attribute:: yBins
      :type: int

      Number of bins on y-axis.

   .. attribute:: xMin
      :type: float

      Minimum value on x-axis.

   .. attribute:: xMax
      :type: float

      Maximum value on x-axis.

   .. attribute:: yMin
      :type: float

      Minimum value on y-axis.

   .. attribute:: yMax
      :type: float

      Maximum value on y-axis.

   .. attribute:: ColorMap
      :type: str

      Colormap name (e.g., "viridis", "plasma", "hot").

   **Methods**

   .. method:: Fill(x: float, y: float, weight: float = 1.0)

      Fill histogram with values.

      :param x: X value
      :param y: Y value
      :param weight: Event weight
      :type x: float
      :type y: float
      :type weight: float

   .. method:: Draw(style: str = "colormesh")

      Draw the 2D histogram.

      :param style: Drawing style: "colormesh", "contour", "contourf"
      :type style: str

   .. method:: SaveFigure(filename: str, dpi: int = 300)

      Save figure to file.

   .. method:: Clear()

      Clear histogram contents.

Advanced Features
-----------------

ATLAS Style Plots
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.plotting import Histogram
   import mplhep
   
   # Use ATLAS style
   mplhep.style.use(mplhep.style.ATLAS)
   
   hist = Histogram()
   hist.Title = "ATLAS Simulation"
   # ... configure and fill histogram ...
   hist.Draw()

Stacked Histograms
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.plotting import Histogram
   import matplotlib.pyplot as plt
   
   # Create multiple histograms
   signal = Histogram()
   background1 = Histogram()
   background2 = Histogram()
   
   # Fill histograms...
   
   # Draw stacked
   fig, ax = plt.subplots()
   signal.Draw()
   background1.Draw()
   background2.Draw()
   
   plt.legend(["Signal", "Background 1", "Background 2"])
   plt.savefig("stacked.png")

Ratio Plots
~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.plotting import Histogram
   import matplotlib.pyplot as plt
   from matplotlib.gridspec import GridSpec
   
   data = Histogram()
   mc = Histogram()
   
   # Fill histograms...
   
   # Create figure with ratio plot
   fig = plt.figure(figsize=(8, 8))
   gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0)
   
   # Main plot
   ax1 = fig.add_subplot(gs[0])
   data.Draw()
   mc.Draw()
   ax1.legend(["Data", "MC"])
   
   # Ratio plot
   ax2 = fig.add_subplot(gs[1], sharex=ax1)
   ratio = data.GetIntegral() / mc.GetIntegral()
   ax2.plot([data.xMin, data.xMax], [ratio, ratio])
   ax2.set_ylabel("Data/MC")
   
   plt.savefig("ratio_plot.png")

Error Bars
~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.plotting import Histogram
   
   hist = Histogram()
   hist.Style = "errorbar"  # Show error bars
   
   # Fill histogram...
   hist.Draw()

Custom Styling
~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.plotting import Histogram
   
   hist = Histogram()
   hist.Color = "blue"
   hist.LineWidth = 2.0
   hist.Alpha = 0.7
   hist.Style = "fill"
   
   # Configure bins
   hist.xBins = 100
   hist.xMin = 0
   hist.xMax = 1000
   
   # Labels with LaTeX
   hist.xTitle = r"$p_T$ [GeV]"
   hist.yTitle = r"$\frac{dN}{dp_T}$ [GeV$^{-1}$]"

Integration with boost-histogram
---------------------------------

The plotting module uses boost-histogram for efficient histogram operations:

.. code-block:: python

   from AnalysisG.core.plotting import Histogram
   import boost_histogram as bh
   
   hist = Histogram()
   # Internal boost-histogram object accessible
   boost_hist = hist._histogram

Performance
-----------

The plotting module is optimized for:

- Fast filling (boost-histogram backend)
- Efficient memory usage
- Large datasets (millions of events)
- Batch plotting operations

See Also
--------

* `boost-histogram documentation <https://boost-histogram.readthedocs.io/>`_
* `mplhep documentation <https://mplhep.readthedocs.io/>`_
* `matplotlib documentation <https://matplotlib.org/>`_
