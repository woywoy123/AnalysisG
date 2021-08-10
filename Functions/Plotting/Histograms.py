import matplotlib.pyplot as plt
from Functions.IO.Files import WriteDirectory
from Functions.Tools.Alerting import Notification
import os

class SharedMethods(WriteDirectory, Notification):
    def __init__(self):
        self.Filename = self.Title + ".png"
        self.DefaultScaling = 8
        self.DefaultDPI = 500
        WriteDirectory.__init__(self)
        Notification.__init__(self)
        self.Caller = "PLOTTING"
        self.Verbose = True

    def SaveFigure(self, dir = ""):
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


class TH1F(SharedMethods):
    def __init__(self, PLT = ""):
        self.Title = ""
        self.xTitle = ""
        self.yTitle = ""
        self.xMin = ""
        self.xMax = ""
        self.yMin = ""
        self.yMax = ""
        self.Bins = ""
        self.Align = "left"
        self.Normalize = True
        self.Data = []

        SharedMethods.__init__(self)

        if PLT == "":
            self.PLT = plt
            self.PLT.figure(figsize=(self.DefaultScaling, self.DefaultScaling), dpi = self.DefaultDPI)

    def CompileHistogram(self):
        self.PLT.title(self.Title)
        if self.xMin == "":
            self.xMin = min(self.Data) - min(self.Data)*0.01
        if self.xMax == "":
            self.xMax = max(self.Data) + max(self.Data)*0.01
        
        self.PLT.hist(self.Data, bins = self.Bins, range=(self.xMin, self.xMax))
        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)

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
        self.AppendToKey("bins", hist.Bins)
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
    



    

