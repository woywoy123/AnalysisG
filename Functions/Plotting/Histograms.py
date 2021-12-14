import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
from Functions.IO.Files import WriteDirectory
from Functions.Tools.Alerting import Notification
import os
import numpy as np


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
        self.Color = "black"
        self.Marker = "o"
        self.Style = ""
        self.Filename = ""

        self.DefaultScaling = 8
        self.DefaultDPI = 500
        self.Compiled = False
        
        self.PLT = plt
        self.PLT.figure(figsize=(self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)

        self.xData = []
        self.yData = []

        self.xLabels = []
        self.yLabels = []

    def xAxis(self):
        if self.xMin == "":
            self.xMin = min(self.xData)

        if self.xMax == "":
            self.xMax = max(self.xData)

        if self.xBins == "":
            self.xBins = int(max(self.xData))

        delta = float(self.xMax - self.xMin)/float(self.xBins)
        self.xMin -= delta*0.5
        self.xMax += delta*0.5
        self.xBins += int(delta)

    def yAxis(self):
        if self.yMin == "":
            self.yMin = min(self.yData)
    
        if self.yMax == "":
            self.yMax = max(self.yData)

        if self.yBins == "":
            self.yBins = int(max(self.yData))

        delta = float(self.yMax - self.yMin)/float(self.yBins)
        self.yMin -= delta*0.5
        self.yMax += delta*0.5
        self.yBins += int(delta)

    def CheckGivenData(self, Data):
        for i in Data:
            if isinstance(i, str):
                return True

    def TransformData(self, Data, Labels):
        nData = getattr(self, Data)
        nLabels = getattr(self, Labels)
        if self.CheckGivenData(nData):
            nLabels = nData.copy()
            nData = []
            for i in range(len(nLabels)):
                nData.append(i)
            setattr(self, Data, nData) 
            setattr(self, Labels, nLabels) 

class SharedMethods(WriteDirectory, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Notification.__init__(self)
        self.Caller = "PLOTTING"
        self.Verbose = True

    def SaveFigure(self, dir = ""):
        
        if self.Compiled != True:
            try:
                self.CompileHistogram()
            except AttributeError:
                pass
       
        if self.Filename == "":
           self.Filename = self.Title + ".png"
        
        self.Notify("SAVING FIGURE AS +-> " + self.Filename)
        if dir == "":
            self.MakeDir("Plots/")
            self.ChangeDir("Plots/")
        else:
            self.MakeDir(dir)
            self.ChangeDir(dir)
         
        if self.Caller != "PLOTTING":
            self.PLT.close("all")
            A = to_agraph(self.G)
            A.layout("dot")

            if ".png" not in self.Filename:
                A.draw(self.Filename + ".png")
            else:
                A.draw(self.Filename)
            self.ChangeDirToRoot()
            return None 

        if ".png" not in self.Filename:
            self.PLT.savefig(self.Filename + ".png")
        else: 
            self.PLT.savefig(self.Filename)
        self.ChangeDirToRoot()
        self.PLT.close()


class TH1F(SharedMethods, GenericAttributes):
    def __init__(self, PLT = ""):

        self.Normalize = False
        self.Log = False

        SharedMethods.__init__(self)
        GenericAttributes.__init__(self)

    def CompileHistogram(self):
         
        self.TransformData("xData", "xLabels")
        self.PLT.title(self.Title)
        self.xAxis()
        
        if len(self.yData) == len(self.xData):
            Update = []
            for i in self.xData: 
                for k in range(self.yData[i]): 
                    Update.append(i)
            self.xData = Update
        self.PLT.hist(self.xData, bins = self.xBins, range=(self.xMin, self.xMax), alpha = self.Alpha, log = self.Log)

        if len(self.xLabels) != 0:
            self.TransformData("xLabels", "xData")
            self.PLT.xticks(self.xLabels, self.xData, rotation = 45, rotation_mode = "anchor", ha = "right")

        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)
        self.Compiled = True

class TH2F(SharedMethods, GenericAttributes):
    
    def __init__(self, PLT = ""):
        self.Normalize = True
        SharedMethods.__init__(self)
        GenericAttributes.__init__(self)
        self.Colorscheme = self.PLT.cm.jet
    
    def CompileHistogram(self):
 
        self.TransformData("xData", "xLabels")
        self.TransformData("yData", "yLabels")

        if len(self.yData) == len(self.xData): 
            if not isinstance(self.yData[0], list):
                pass
            else:
                yUpdate = []
                xUpdate = []
                for i in self.xData:
                    for j in self.yData[i]: 
                        yUpdate.append(j)
                        xUpdate.append(i)
                self.xData = xUpdate
                self.yData = yUpdate

        self.PLT.title(self.Title)
        self.xAxis()
        self.yAxis()

        self.PLT.hist2d(self.xData, self.yData, bins = [self.xBins, self.yBins], range=[[self.xMin, self.xMax], [self.yMin, self.yMax]], cmap = self.Colorscheme)
        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)
        self.PLT.colorbar()

        if len(self.xLabels) != 0:
            self.TransformData("xLabels", "xData")
            self.PLT.xticks(self.xLabels, self.xData, rotation = 45, rotation_mode = "anchor", ha = "right")
        if len(self.yLabels) != 0:
            self.PLT.yticks(self.yLabels, self.yData, rotation = 45, rotation_mode = "anchor", ha = "right")
        self.Compiled = True

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
        self.Compiled = False
        self.PLT.figure(figsize=(self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)

    def CompileHistogram(self):

        if self.Title != "":
            self.PLT.title(self.Title)
        
        for i in range(len(self.Histograms)):
            H = self.Histograms[i]
            self.PLT.hist(H.xData, bins = H.xBins, range=(H.xMin,H.xMax), label = H.Title, alpha = H.Alpha, log = self.Log)
        
        if len(self.Histograms) != 0:
            self.PLT.xlabel(self.Histograms[0].xTitle)
            self.PLT.ylabel(self.Histograms[0].yTitle)
        self.PLT.legend(loc="upper right")
        if self.Filename == "":
            self.Filename = self.Title + ".png"
        self.Compiled = True

    def Save(self, dir):
        self.SaveFigure(dir) 


class CombineTGraph(SharedMethods):
    
    def __init__(self):
        SharedMethods.__init__(self)
        self.DefaultScaling = 8
        self.DefaultDPI = 500
        self.Normalize = False
        self.Lines = []
        self.Title = ""
        self.Log = False
        self.PLT = plt
        self.PLT.figure(figsize=(self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)
        self.yMin = 0
        self.yMax = 1
    
    def CompileLine(self):
        if self.Title != "":
            self.PLT.title(self.Title)

        for i in range(len(self.Lines)):
            H = self.Lines[i]

            if H.ErrorBars:
                self.PLT.errorbar(x = H.xData, y = H.yData, yerr = [H.Lo_err, H.Up_err], color = H.Color, linestyle = "-", capsize = 3, linewidth = 1, alpha = H.Alpha, label = H.Title)
            else:
                self.PLT.plot(H.xData, H.yData, marker = H.Marker, color = H.Color, linewidth = 1, alpha = H.Alpha, label = H.Title)
        
        self.PLT.ylim(self.yMin, self.yMax)

        if len(self.Lines) != 0:
            self.PLT.xlabel(self.Lines[0].xTitle)
            self.PLT.ylabel(self.Lines[0].yTitle)
        self.PLT.legend(loc="upper right")
        self.Filename = self.Title + ".png"

    def Save(self, dir):
        self.SaveFigure(dir) 







class SubfigureCanvas(SharedMethods):
    def __init__(self):
        self.FigureObjects = []
        self.Title = ""
        self.DefaultScaling = 8
        self.DefaultDPI = 500
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
        self.PLT.hist(hist.xData, **self.__dic)
        self.PLT.xlabel(hist.xTitle)
        self.PLT.ylabel(hist.yTitle)       
    
    def CompileHistogram(self):
        self.PLT = plt
        self.PLT.figure(figsize = (len(self.FigureObjects)*self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)
        
        self.y = 1
        self.x = len(self.FigureObjects)
        self.k = 0
        for i in self.FigureObjects:
            self.k += 1
            self.AppendToPLT(i)

class TGraph(SharedMethods, GenericAttributes):

    def __init__(self):
        SharedMethods.__init__(self)
        GenericAttributes.__init__(self)
        self.ErrorBars = False

    def Line(self):
        err = []
        self.Up_err = []
        self.Lo_err = []
        if isinstance(self.yData[0], list) == True:
            self.ErrorBars = True
            tmp = []
            for i in self.yData:
                av = np.array(i)
                tmp.append(np.average(av))
                err.append(np.std(av))  
                self.Up_err.append(abs(np.average(av)+np.std(av)))
                self.Lo_err.append(abs(np.average(av)-np.std(av)))
                
            self.yData = tmp
        
        self.yBins = 1
        self.xAxis()
        self.yAxis()

        if self.ErrorBars:
            self.PLT.errorbar(x = self.xData, y = self.yData, yerr = [self.Lo_err, self.Up_err], color = self.Color, linestyle = "-", capsize = 3, linewidth = 1)
        else:
            self.PLT.plot(self.xData, self.yData, marker = self.Marker, color = self.Color, linewidth = 1)
        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)
        self.PLT.xlim(0, self.xMax)
        self.PLT.ylim(0, self.yMax*2)
    
