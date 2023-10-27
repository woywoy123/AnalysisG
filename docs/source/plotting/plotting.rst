Plotting (TH1F, TH2F, CombineTH1F)
**********************************

A class dedicated to plotting histograms using the `mplhep` package as a backend to format figures.
This class adds some additional features to simplify writing simple plotting code, such as bin centering. 


.. py:class:: AnalysisG.Plotting.BasePlotting

    .. py:function:: __precompile__(self)

        A function which can be overridden and is used to perform preliminary data manipulation or histogram modifications.

    .. py:function:: __compile__(self)

    .. py:function:: __postcompile__(self)

    .. py:attribute:: SaveFigure(self, dirc)

        Whether to compile the given histogram object. 
        
        :params str dirc: The output directory of the plotting.


    .. py:attribute:: OutputDirectory -> str

        The directory in which to save the figure. 
        If the directory tree is non-existent, it will automatically be created.

    .. py:attribute:: Filename -> str
   
        The name given to the output `.png` file.

    .. py:attribute:: DPI -> int 

        The resolution of the figure to save. 
    
    .. py:attribute:: FontSize -> int

        The front size to use for text on the plot.
    
    .. py:attribute:: LabelSize -> int 

        Ajusts the label sizes on the plot.
    
    .. py:attribute:: LaTeX -> bool

        Whether to use the *LaTeX* engine of `MatplotLib`
    
    .. py:attribute:: TitleSize -> int 

        Modify the title font size.
    
    .. py:attribute:: LegendSize -> int
   
        Modify the size of the legend being displayed on the plot.
        This is predominantly relevant for combining `TH1F` histograms.



    .. py:attribute:: LegendLoc
    
    .. py:attribute:: NEvents -> int 

        Displays the number of events used to construct the histogram. 
    
    .. py:attribute:: xScaling -> float 

        A scaling multiplier in the x-direction of the plot.
        This is useful when bin labels start to merge together.
    
    .. py:attribute:: yScaling -> float 

        A scaling multiplier in the y-direction of the plot.
        This is useful when bin labels start to merge together.
    
    .. py:attribute:: Alpha -> float

        The alpha by which the color should be scaled by. 
    
    .. py:attribute:: LineWidth
    
    .. py:attribute:: HistFill

        Whether to fill the histograms with the color assigned or not.
    
    .. py:attribute:: Color -> str

        The color to assign the histogram.

    .. py:attribute:: Colors

        Expects a list of string indicating the color each histogram should be assigned.
        If none is given then colors will be automatically assigned to each histogram.

    .. py:attribute:: Texture
        
        The filling pattern of the histogram, options are; 
        / , \\ , | , - , + , x, o, O, ., '*', True, False

    .. py:attribute:: Marker

    .. py:attribute:: Textures

    .. py:attribute:: Markers

    .. py:attribute:: autoscale

    .. py:attribute:: xLogarithmic

        Whether to scale the bin content logarithmically.

    .. py:attribute:: yLogarithmic

        Whether to scale the bin content logarithmically.

    .. py:attribute:: Title -> str

        Main title of the histogram.

    .. py:attribute:: xTitle -> str

        Title to place on the x-Axis.

    .. py:attribute:: xBinCentering -> bool

        Whether to center the bins of the histograms. 
        This can be relevant for classification plots.

    .. py:attribute:: xBins -> int 

        The number of bins to construct the histogram with.

    .. py:attribute:: yTitle -> str

        Title to place on the y-Axis.

    .. py:attribute:: Style -> str

        The style to use for plotting the histogram, options are:
        - "ATLAS"
        - "ROOT"
        - "MPL"

    .. py:attribute:: xMin -> float, int

        The minimum value to start the x-Axis with.

    .. py:attribute:: xMax -> float, int

        The maximum value to end the x-Axis with.

    .. py:attribute:: yMin -> float, int

        The minimum value to start the y-Axis with.

    .. py:attribute:: yMax -> float, int

        The maximum value to end the y-Axis with.

    .. py:attribute:: xStep -> float, int

        The step size of placing a label on the x-Axis, e.g. 0, 100, 200, ..., (n-1)x100.

    .. py:attribute:: yStep -> float, int

        The step size of placing a label on the y-Axis, e.g. 0, 100, 200, ..., (n-1)x100.

    .. py:attribute:: xData(self)

    .. py:attribute:: yData(self):

    .. py:attribute:: xLabels

        A list of string/values to place on the x-Axis for each bin. 
        The labels will be placed in the same order as given in the list.

    .. py:attribute:: yLabels

        A list of string/values to place on the y-Axis for each bin. 
        The labels will be placed in the same order as given in the list.

    .. py:attribute:: ATLASLumi

        The luminosity to display on the `ATLAS` formated histograms. 

    .. py:attribute:: ATLASData -> bool

        A boolean switch to distinguish between *Simulation* and *Data*.

    .. py:attribute:: ATLASYear -> int

        The year the data/simulation was collected from.

    .. py:attribute:: ATLASCom -> float

        The *Center of Mass* used for the data/simulation.



.. py:class:: AnalysisG.Plotting.TH1F(AnalysisG.Plotting.BasePlotting):

    A simple histogram plotting class used to minimize redundant styling code.
    The class can be further adapted in a custom framework via class inheritance.

    .. py:function:: __histapply__()

    .. py:function:: __makelabelaxis__()

    .. py:function:: __fixrange__()

    .. py:function:: __aggregate__()

    .. py:function:: __precompile__()

    .. py:function:: __inherit_compile__(inpt)

    .. py:function:: __compile__()

    .. py:function:: __postcompile__()

    .. py:attribute:: Histogram -> TH1F

        A single `TH1F` object used to plot against (useful for underlying distributions).

    .. py:attribute:: Histograms -> list[TH1F]
   
        Expects `TH1F` objects from which to construct the combined histogram.

    .. py:attribute:: Stack -> bool   

        Whether to combine the histograms as a stack plot.


.. py:class:: AnalysisG.Plotting.TH2F(AnalysisG.Plotting.BasePlotting):

    .. py:function:: __histapply__()

    .. py:function:: __makelabelaxis__()

    .. py:function:: __fix_xrange__()

    .. py:function:: __fix_yrange__()

    .. py:function:: __precompile__()

    .. py:function:: __compile__():

    .. py:function:: __postcompile__():

    .. py:attribute:: yBins -> int

    .. py:attribute:: xBins -> int

    .. py:attribute:: xUnderFlow -> bool

    .. py:attribute:: yUnderFlow -> bool

    .. py:attribute:: xOverFlow -> bool
   
    .. py:attribute:: yOverFlow -> bool



.. py:class:: AnalysisG.Plotting.TLine(BasePlotting):

    .. py:function:: __fixrange__()

    .. py:function:: __lineapply__()

    .. py:function:: __precompile__()

    .. py:function:: __compile__()

    .. py:function:: __postcompile__()


    .. py:attribute:: yDataUp -> list[float, int, ...]

    .. py:attribute:: yDataDown -> list[float, int, ...]








Attributes (Data)
_________________

- ``xData``:
  The data from which to construct the histogram. 
  If this is to be used with `xTickLabels`, make sure the bin numbers are mapped to the input list.
  For example; `xData = [0, 1, 2, 3, 4]  -> xTickLabels = ["b1", "b2", "b3", "b4", "b5"]`

- ``xWeights``:
  Weights to be used to scale the bin content. 
  This is particularly useful for using `xTickLabels`.

- ``Normalize``:
  Whether to normalize the data. Options are; `%`, `True` or `False`.


Example: A simple TH1F plot
___________________________

.. code-block:: python 

    from AnalysisG.Plotting import TH1F

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


Example: A TH1F plot with bin centering 
_______________________________________

.. code-block:: python 

    from AnalysisG.Plotting import TH1F

    th = TH1F()
    th.xMin = 0
    th.xStep = 20
    #th.xMax = 100 <- dont include a maximum
    th.xBins = 100 # <- rather define the number of bins
    th.xBinCentering = True
    th.xData = [i for i in range(100)]
    th.Title = "some title"
    th.xTitle = "x-Axis"
    th.yTitle = "y-Axis"
    th.Filename = "some-name"
    th.OutputDirectory = "./Some/Path/"
    th.SaveFigure()

Example: Combining two or more TH1F plots 
_________________________________________

.. code-block:: python 

    from AnalysisG.Plotting import TH1F

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


    # To make the above code shorter, we can create a dictionary
    # of commands e.g. 
    tmp = {"xMin" : 0, "xStep" : 20, ... , "Histograms" : []}

    # and then do the same loop over the data, but populate the Histograms 
    # key in the tmp dictionary 

    for i in MyDataDictionary:
        tmp2 = {"xData" : MyDataDictionary[i], "Title" : i}
        tmp["Histograms"].append(TH1F(**tmp2))

    th = TH1F(**tmp)
    th.SaveFigure()


Example: A Simple TH2F Plot
___________________________

.. code-block:: python 

    from AnalysisG.Plotting import TH2F

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

