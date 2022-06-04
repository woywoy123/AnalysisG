from Functions.IO.Files import WriteDirectory
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.Event.CacheGenerators import Generate_Cache
from Functions.Plotting.Histograms import TH2F, TH1F, CombineHistograms

def CreateWorkspace(Name, Dir, Cache, Stop = 100):
    if Cache:
        x = WriteDirectory()
        x.MakeDir("PresentationPlots/" + Name)
        Generate_Cache(Dir, Stop = Stop, Outdir = "PresentationPlots/" + Name)
    return UnpickleObject("EventGenerator", "PresentationPlots/" + Name + "/EventGenerator")
     
def Histograms_Template(Title, xTitle, yTitle, bins, Min, Max, Data, FileName = None, Dir = None, Color = None, Weight = None, Alpha = 0.25, DPI = None):
    H = TH1F()
    H.Title = Title
    H.xTitle = xTitle
    H.yTitle = yTitle
    H.xBins = bins
    H.xMin = Min
    H.xMax = Max
    H.xData = Data
    H.Alpha = Alpha
    H.Weights = Weight
    H.DefaultScaling = 7

    if DPI != None:
        H.DefaultDPI = DPI
    
    if Color is not None:
        H.Color = Color

    if FileName != None:
        H.Filename = FileName
        H.SaveFigure(Dir)
    return H

def Histograms2D_Template(Title, xTitle, yTitle, xBins, yBins, xMin, xMax, yMin, yMax, xData, yData, FileName, Dir, Diagonal = False, Weight = None):
    H = TH2F()
    H.Diagonal = Diagonal
    H.Title = Title
    H.xTitle = xTitle
    H.yTitle = yTitle
    H.xBins = xBins 
    H.yBins = yBins 
    H.xMin = xMin
    H.xMax = xMax
    H.yMin = yMin
    H.yMax = yMax
    H.xData = xData
    H.yData = yData
    H.Weights = Weight
    H.ShowBinContent = True

    H.Filename = FileName
    H.SaveFigure("Plots/" + Dir)
    return H


def HistogramCombineTemplate(DPI = 500, Scaling = 7, Size = 10):
    T = CombineHistograms()
    T.DefaultDPI = DPI
    T.DefaultScaling = Scaling
    T.LabelSize = Size + 5
    T.FontSize = Size
    T.LegendSize = Size
    return T
