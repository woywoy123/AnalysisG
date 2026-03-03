Plotting (Python)
=================

The ``BasePlotting``, ``TH1F``, ``TH2F``, and ``TLine`` Cython classes
wrap the C++ ``plotting`` engine.  They use ``matplotlib`` under the hood
and support LaTeX labels, stacked histograms, ratio panels, and
KS-test annotations.

.. note::
   ``BasePlotting`` cannot be instantiated directly.  Use ``TH1F``,
   ``TH2F``, or ``TLine``.

BasePlotting — Common Properties
---------------------------------

All plotting classes inherit the following properties.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``Filename``
     - ``str``
     - Output file stem (extension added automatically).
   * - ``OutputDirectory``
     - ``str``
     - Directory where the figure is saved.
   * - ``Title``
     - ``str``
     - Figure title (rendered as LaTeX if ``UseLateX=True``).
   * - ``xTitle``
     - ``str``
     - x-axis label.
   * - ``yTitle``
     - ``str``
     - y-axis label.
   * - ``Style``
     - ``str``
     - matplotlib style sheet name (e.g. ``"default"``, ``"ggplot"``).
   * - ``DPI``
     - ``int``
     - Figure resolution in dots per inch.
   * - ``FontSize``
     - ``float``
     - Global font size.
   * - ``AxisSize``
     - ``float``
     - Axis label font size.
   * - ``LegendSize``
     - ``float``
     - Legend font size.
   * - ``TitleSize``
     - ``float``
     - Title font size.
   * - ``UseLateX``
     - ``bool``
     - Render all text with LaTeX.
   * - ``xScaling``
     - ``float``
     - Figure width scaling factor.
   * - ``yScaling``
     - ``float``
     - Figure height scaling factor.
   * - ``AutoScaling``
     - ``bool``
     - Automatically scale figure dimensions.
   * - ``LineStyle``
     - ``str``
     - matplotlib line style string (e.g. ``"-"``, ``"--"``).
   * - ``xLogarithmic``
     - ``bool``
     - Use log scale on the x-axis.
   * - ``yLogarithmic``
     - ``bool``
     - Use log scale on the y-axis.
   * - ``xStep``
     - ``float``
     - x-axis tick step.
   * - ``yStep``
     - ``float``
     - y-axis tick step.
   * - ``xMin / xMax``
     - ``float``
     - x-axis range.
   * - ``yMin / yMax``
     - ``float``
     - y-axis range.
   * - ``Overflow``
     - ``bool``
     - Accumulate entries beyond ``xMax`` into the last bin.
   * - ``Color``
     - ``str``
     - Single colour for the current dataset.
   * - ``Colors``
     - ``list[str]``
     - Colour list when plotting multiple datasets.
   * - ``Hatch``
     - ``str``
     - matplotlib hatch pattern (e.g. ``"///"``, ``"xxx"``).
   * - ``CapSize``
     - ``float``
     - Error-bar cap size.
   * - ``ErrorBars``
     - ``bool``
     - Draw error bars on the histogram.

TH1F — 1D Histogram
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``xData``
     - ``list[float]``
     - Raw data values to histogram.
   * - ``xBins``
     - ``int``
     - Number of histogram bins.
   * - ``Weights``
     - ``list[float]``
     - Per-entry weights (same length as ``xData``).
   * - ``CrossSection``
     - ``float``
     - Sample cross-section used for normalisation.
   * - ``IntegratedLuminosity``
     - ``float``
     - Luminosity used for cross-section rescaling.
   * - ``HistFill``
     - ``bool``
     - Fill histogram area with colour.
   * - ``Stacked``
     - ``bool``
     - Stack multiple datasets vertically.
   * - ``LineWidth``
     - ``float``
     - Histogram outline width.
   * - ``Alpha``
     - ``float``
     - Fill opacity (0–1).
   * - ``Density``
     - ``bool``
     - Normalise histogram to unit area.
   * - ``Marker``
     - ``str``
     - matplotlib marker style.
   * - ``xLabels``
     - ``dict``
     - Map of bin-centre value → label string for categorical axes.
   * - ``ShowCount``
     - ``bool``
     - Annotate each bin with its count.

TH2F — 2D Histogram
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``xData``
     - ``list[float]``
     - x-axis data values.
   * - ``yData``
     - ``list[float]``
     - y-axis data values.
   * - ``xBins``
     - ``int``
     - Number of x-axis bins.
   * - ``yBins``
     - ``int``
     - Number of y-axis bins.
   * - ``xLabels``
     - ``dict``
     - Categorical labels for x-axis bins.
   * - ``yLabels``
     - ``dict``
     - Categorical labels for y-axis bins.
   * - ``Weights``
     - ``list[float]``
     - Per-entry weights.

TLine — Line Plot
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``xData``
     - ``list[float]``
     - x-axis values.
   * - ``yData``
     - ``list[float]``
     - y-axis values.
   * - ``yDataUp``
     - ``list[float]``
     - Upper uncertainty band values.
   * - ``yDataDown``
     - ``list[float]``
     - Lower uncertainty band values.
   * - ``Marker``
     - ``str``
     - matplotlib marker style.
   * - ``LineWidth``
     - ``float``
     - Line width.
   * - ``Alpha``
     - ``float``
     - Line opacity.
