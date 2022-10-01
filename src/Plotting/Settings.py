import mplhep as hep
import math

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

import random

from AnalysisTopGNN.IO import WriteDirectory
from AnalysisTopGNN.Tools import Notification


class Settings:
    def __init__(self):
        self.Cosmetics()
        self.Layout()
        self.IO()
        self.ResetPLT()

    def Cosmetics(self):
        self.Style = None
        self.ATLASData = False
        self.ATLASYear = None
        self.ATLASLumi = None
        self.ATLASCom = None
        self.Color = None
        
    def Layout(self):
        self.FontSize = 10
        self.LabelSize = 12.5
        self.TitleSize = 10
        self.LegendSize = 10

        self.Logarithmic = False
        self.Scaling = 1.25
        self.DPI = 250
    
    def IO(self):
        self.Title = None
        self.Filename = None
        self.OutputDirectory = "Plots"

    def DefineStyle(self):
        if self.Style == "ATLAS":
            hep.atlas.text(loc = 2)
            hep.atlas.label(data = self.ATLASData, 
                    year = self.ATLASYear, 
                    lumi = self.ATLASLumi, 
                    com = self.ATLASCom)
            self.PLT.style.use(hep.style.ATLAS)

        if self.Style == "ROOT":
            self.PLT.style.use(hep.style.ROOT)

        if self.Style == None:
            pass
    
    def MakeFigure(self): 
        self.Figure, self.Axis = plt.subplots(figsize = (self.Scaling*6.4, self.Scaling*4.8))
        self.Axis.set_autoscale_on(True)
    
    def ApplyToPLT(self):
        self.PLT.rcParams.update({
            "font.size":self.FontSize, 
            "axes.labelsize" : self.LabelSize, 
            "legend.fontsize" : self.LegendSize, 
            "figure.titlesize" : self.TitleSize
            })
        self.PLT.rcParams["text.usetex"] = True
    
    def ResetPLT(self):
        plt.close("all")
        self.PLT = plt
        self.PLT.rcdefaults()
        self.MakeFigure()

class CommonFunctions(WriteDirectory, Settings, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Settings.__init__(self)
        Notification.__init__(self)
    
    def DefineAxisData(self, Dim, JustData = False):
        self.Set(Dim + "Data", [])
        if JustData: 
            return 
        self.Set(Dim + "Min", None)
        self.Set(Dim + "Max", None)
        self.Set(Dim + "Weights", None)
        self.Set(Dim + "Title", None)
        self.Set(Dim + "TickLabels", None)

    def ApplyInput(self, args):
        for key, val in args.items():
            if key not in self.__dict__:
                self.Warning("Provided variable " + key + " not found")
                continue
            self.__dict__[key] = val

    def Get(self, var):
        return getattr(self, var)

    def Set(self, var, val):
        setattr(self, var, val)

    def SanitizeData(self, var):
        out = []
        for i in range(len(var)):
            if var[i] == None:
                continue
            if math.isinf(var[i]): 
                continue
            if math.isnan(var[i]):
                continue
            out.append(var[i]) 
        return out 

    def DefineCommonRange(self, Dims):
        
        x = self.SanitizeData(self.Get(Dims + "Data"))
        if len(x) == 0:
            self.Warning("NO VALID DATA GIVEN ... Skipping: " + self.Title)
            return 
        self.Set(Dims + "Data", x)

        if self.Get(Dims + "Min") == None:
            self.Set(Dims + "Min", min(self.Get(Dims + "Data")))

        if self.Get(Dims + "Max") == None:
            self.Set(Dims + "Max", max(self.Get(Dims + "Data")))
   
    def ApplyRandomColor(self, obj):
        color = next(self.Axis._get_lines.prop_cycler)["color"]
        if obj.Color == None:
            obj.Color = color
            self.Colors.append(color)
        elif obj.Color in self.Colors:
            obj.Color = None
            self.ApplyRandomColor(obj)
            return 

    def Precompiler(self):
        pass

    def SaveFigure(self, Dir = None):
        self.Precompiler()
        if Dir == None:
            Dir = self.OutputDirectory
        if Dir.endswith("/") == False:
            Dir += "/"
        if self.Filename.endswith(".png") == False:
            self.Filename += ".png"
    
        self.MakeDir(Dir)
        self.ChangeDir(Dir)
        
        self.Compile()
        
        self.Axis.set_title(self.Title)
        self.PLT.xlabel(self.xTitle, size = self.LabelSize)
        self.PLT.ylabel(self.yTitle, size = self.LabelSize)

        self.PLT.tight_layout()
        self.PLT.savefig(self.Filename, dpi = self.DPI)
        self.ChangeDirToRoot()
        self.PLT.close("all")

        self.Notify("SAVING FIGURE AS +-> " + Dir + self.Filename)
