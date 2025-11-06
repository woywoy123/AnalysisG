.. _plotting:

Plotting 
--------

The plotting class is composed of the `TH1F`, `TH2F` and `TLine` classes. 
These names are taken from the `CERN ROOT` framework and are used as wrappers for `boost-histograms` and `MPL-HEPP`. 

Plotting Base Function used to Share Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: AnalysisG.core.plotting.BasePlotting

   :ivar str Hatch: Changes the texture of figures.

   :ivar int DPI: Changes the output resolution of the figure.

   :ivar str Style: The template style to use (ATLAS, ROOT, etc.)

   :ivar bool ErrorBars: Indicates whether to show error bars on the histogram bins.

   :ivar str Filename: Specifies the output filename for the plot.

   :ivar str OutputDirectory: Specifies the output directory for the plot.

   :ivar float FontSize: Specifies the font size to be used by matplotlib.

   :ivar float AxisSize: Specifies the axis-title font size.

   :ivar float LegendSize: Specifies the size of the legend shown on plots.

   :ivar float TitleSize: Specifies the title size shown on plots.

   :ivar bool UseLateX: Whether to parse mathematical symbols via latex.

   :ivar float xScaling: Specify the image scaling in the x-direction.

   :ivar float yScaling: Specify the image scaling in the y-direction.

   :ivar bool AutoScaling: Whether to use matplot lib's magic to find the optimal scaling.

   :ivar str Title: The title given for the plot.

   :ivar str xTitle: The title given to the x-axis.

   :ivar str yTitle: The title given to the y-axis.

   :ivar bool xLogarithmic: Whether to scale the x-axis logarithmically.

   :ivar bool yLogarithmic: Whether to scale the y-axis logarithmically.

   :ivar float xStep: Specifies the step size of the x-axis ticks.

   :ivar float yStep: Specifies the step size of the y-axis ticks.

   :ivar float xMin: Specifies the minimum value of the plots x-axis.

   :ivar float yMin: Specifies the minimum value of the plots y-axis.

   :ivar float xMax: Specifies the maximum value of the plots x-axis.

   :ivar float yMax: Specifies the maximum value of the plots y-axis.

   .. py:function:: SaveFigure

      A function call used to save the figure.



Histogramming using TH1F
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: AnalysisG.core.plotting.TH1F(BasePlotting)

   :ivar list[float] xData: Assigns the values that should be used to fill the histogram.

   :ivar int xBins: Number of bins to use for the histogram.

   :ivar float CrossSection: A variable used to specify the cross section of the process, which scales the histogram accordingly.

   :ivar float IntegratedLuminosity: The integrated luminosity used to scale the histogram.

   :ivar str HistFill: A parameter used to modify the fill type of the histogram.

   :ivar bool Stacked: Whether to plot a stacked histogram.

   :ivar float LineWidth: Specifies the line width of the histograms.

   :ivar float Alpha: Controls the transparency of the histograms.

   :ivar bool Density: Whether to plot the histogram as a density.

   :ivar dict xLabels: Labels to apply on the x-axis with specified weight values.


Histogramming using TH2F
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: AnalysisG.core.plotting.TH2F(BasePlotting)

   :ivar list[float] xData: Assigns the values that should be used to fill along the x-axis.

   :ivar list[float] yData: Assigns the values that should be used to fill along the y-axis.

   :ivar int xBins: Number of bins to use along the x-axis.

   :ivar int yBins: Number of bins to use along the x-axis.





Example: A simple TH1F plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 

    from AnalysisG.core.plotting import TH1F

    th = TH1F()
    th.xBins = 100
    th.xMax = 100
    th.xMin = 0
    th.xData = [i for i in range(100)]
    th.Title = "some title"
    th.xTitle = "x-Axis"
    th.yTitle = "y-Axis"
    th.Filename = "some-name"
    th.OutputDirectory = "./Some/Path/"
    th.SaveFigure()


Example: Combining two or more TH1F plots 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 

    from AnalysisG.core.plotting import TH1F

    # Define the settings to apply to all histograms
    th = TH1F()
    th.xMin = 0
    th.xStep = 20
    th.xMax = 100
    th.Title = "some title"
    th.xTitle = "x-Axis"
    th.yTitle = "y-Axis"
    th.Filename = "some-name"
    th.OutputDirectory = "./Some/Path/"

    # Iterate over your data
    for i in MyDataDictionary:

        # Create a new TH1F instance
        th_ = TH1F()
        th_.Title = i

        # Populate this instance with some data
        th_.xData = MyDataDictionary[i]

        # Append the instance to the Histograms attribute
        th.Histograms.append(th_)

    th.SaveFigure()


Example: A Simple TH2F Plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 

    from AnalysisG.core.plotting import TH2F

    th2 = TH2F()
    th2.Title = "Some distribution plot"
    th2.xTitle = "x-Title"
    th2.yTitle = "y-Title"

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 100
    th2.yMax = 100

    th2.xBins = 100
    th2.yBins = 100

    th2.xData = [i for i in range(100)]
    th2.yData = [i for i in range(100)]
    th2.Filename = "Some_File"
    th2.OutputDirectory = "./some/path"
    th2.SaveFigure()

