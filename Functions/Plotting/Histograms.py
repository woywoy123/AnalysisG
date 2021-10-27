import matplotlib.pyplot as plt
from Functions.IO.Files import WriteDirectory
from Functions.Tools.Alerting import Notification
import os

class GenericAttributes:
    def __init__(self):
        self.Title = ""
        self.xTitle = ""
        self.yTitle = ""
        self.zTitle = ""
        
        self.xMin = ""
        self.yMin = ""
        self.zMin = ""
        
        self.xMax = ""
        self.yMax = ""
        self.zMax = ""
        
        self.xBins = ""
        self.yBins = ""
        self.Align = "left"
        
        self.Alpha = 0.5

        self.DefaultScaling = 8
        self.DefaultDPI = 500
        
        self.PLT = plt
        self.PLT.figure(figsize=(self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)

        self.xData = []
        self.yData = []
    
    def minAxis(self, Data, axis):
        if self.xMin == "":
            self.xMin = min(Data) - min(Data)*0.01
    
    def maxAxis(self, Data, axis):
        if self.xMax == "":
            self.xMax = max(Data) + max(Data)*0.01
    
    def xAxis(self):
        self.minAxis(self.xData, self.xMin)
        self.maxAxis(self.xData, self.xMax)

    def yAxis(self):
        self.minAxis(self.yData, self.yMin)
        self.maxAxis(self.yData, self.yMax)


class SharedMethods(WriteDirectory, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Notification.__init__(self)
        self.Caller = "PLOTTING"
        self.Verbose = True

    def SaveFigure(self, dir = ""):
        
        self.CompileHistogram()

        self.Notify("SAVING FIGURE AS +-> " + self.Filename)
        if dir == "":
            self.MakeDir("Plots/")
            self.ChangeDir("Plots/")
        else:
            self.MakeDir(dir)
            self.ChangeDir(dir)
        
        if ".png" not in self.Filename:
            self.PLT.savefig(self.Filename + ".png")
        else: 
            self.PLT.savefig(self.Filename)
        self.ChangeDirToRoot()
        self.PLT.close("all")


class TH1F(SharedMethods, GenericAttributes):
    def __init__(self, PLT = ""):

        self.Normalize = True
        self.Log = False

        SharedMethods.__init__(self)
        GenericAttributes.__init__(self)

    def CompileHistogram(self):
        self.PLT.title(self.Title)
        self.xAxis()
        
        self.PLT.hist(self.xData, bins = self.xBins, range=(self.xMin, self.xMax), alpha = self.Alpha, log = self.Log)
        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)
        self.Filename = self.Title + ".png"

class TH2F(SharedMethods, GenericAttributes):
    
    def __init__(self, PLT = ""):
        self.Normalize = True
        SharedMethods.__init__(self)
        GenericAttributes.__init__(self)
        self.Colorscheme = self.PLT.cm.jet
    
    def CompileHistogram(self):
        self.PLT.title(self.Title)
        self.xAxis()
        self.yAxis()
        
        self.PLT.hist2d(self.xData, self.yData, bins = [self.xBins, self.yBins], range=[[self.xMin, self.xMax], [self.yMin, self.yMax]], cmap = self.Colorscheme)
        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)
        self.PLT.colorbar()
        self.Filename = self.Title + ".png"
       

class CombineHistograms(SharedMethods):
    def __init__(self):
        SharedMethods.__init__(self)
        self.DefaultScaling = 8
        self.DefaultDPI = 500
        self.Normalize = False
        self.Histograms = []
        self.Title = ""
        self.Log = False
        self.PLT = plt
        self.PLT.figure(figsize=(self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)

    def CompileHistogram(self):

        if self.Title != "":
            self.PLT.title(self.Title)

        for i in range(len(self.Histograms)):
            H = self.Histograms[i]
            self.PLT.hist(H.xData, bins = H.xBins, range=(H.xMin,H.xMax), label = H.Title, alpha = self.Alpha, log = self.Log)
        
        if len(self.Histograms) != 0:
            self.PLT.xlabel(self.Histograms[0].xTitle)
            self.PLT.ylabel(self.Histograms[0].yTitle)
        self.PLT.legend(loc="upper right")
        self.Filename = self.Title + ".png"

    def Save(self, dir):
        self.SaveFigure(dir) 

class SubfigureCanvas(SharedMethods):
    def __init__(self):
        self.FigureObjects = []
        self.Title = ""
        SharedMethods.__init__(self)

    def AddObject(self, obj):
        self.FigureObjects.append(obj)
    
    def AppendToKey(self, key, val):
        if val != "":
            self.__dic[key] = val

    def AppendToPLT(self, hist):
        self.__dic = {}
        self.AppendToKey("align", hist.Align)
        self.AppendToKey("bins", hist.xBins)
        self.AppendToKey("range", (hist.xMin, hist.xMax))
        self.AppendToKey("density", hist.Normalize)
        self.AppendToKey("align", hist.Align)
        
        self.PLT.subplot(int(str(self.y) + str(self.x) + str(self.k)))
        self.PLT.title(hist.Title)
        self.PLT.hist(hist.Data, **self.__dic)
        self.PLT.xlabel(hist.xTitle)
        self.PLT.ylabel(hist.yTitle)       
    
    def CompileFigure(self):
        self.PLT = plt
        self.PLT.figure(figsize = (len(self.FigureObjects)*self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)
        
        self.y = 1
        self.x = len(self.FigureObjects)
        self.k = 0
        for i in self.FigureObjects:
            self.k += 1
            self.AppendToPLT(i)
    



    

