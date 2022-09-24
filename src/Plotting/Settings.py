import mplhep as hep
import math

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

import random

class Settings:
    def __init__(self):
        
        # --- Element Sizes ---
        self.FontSize = 10
        self.LabelSize = 12.5
        self.TitleSize = 10
        self.LegendSize = 10

        # --- Figure Settings --- #
        self.Scaling = 1.25
        self.DPI = 250
        
        # --- Histogram Cosmetic Styles --- #
        self.Style = None
        self.RandomTexture = False
        self.Alpha = 0.5
        self.Color = None
        self.FillHist = "fill"
        
        # --- Data Display --- #
        self.Normalize = None
        self.Logarithmic = None

        # --- Properties --- #
        self.Title = None
        self.Filename = None
        self.OutputDirectory = "Plots"
        self.ATLASData = False
        self.ATLASYear = None
        self.ATLASLumi = None
        self.ATLASCom = None
        
        self.ResetPLT()
 
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

class CommonFunctions:
    def __init__(self):
        pass
    
    def DefineAxisData(self, Dim):
        self.Set(Dim + "Data", [])
        self.Set(Dim + "Bins", None)
        self.Set(Dim + "Weights", None)

    def DefineAxis(self, Dim):
        self.Set(Dim + "Min", None)
        self.Set(Dim + "Max", None)
        self.Set(Dim + "Title", None)
        self.Set(Dim + "BinCentering", False)
        self.Set(Dim + "Range", None)

    def ApplyInput(self, args):

        for key, val in args.items():
            if key not in self.__dict__:
                self.Warning("Provided variable " + key + " not found")
                continue
            self.__dict__[key] = val

    def SaveFigure(self, Dir = None):
        if Dir == None:
            Dir = self.OutputDirectory
        if Dir.endswith("/") == False:
            Dir += "/"
        if self.Filename.endswith(".png") == False:
            self.Filename += ".png"
    
        self.MakeDir(Dir)
        self.ChangeDir(Dir)
        self.Notify("SAVING FIGURE AS +-> " + Dir + self.Filename)
        
        self.Compile()
        
        self.Axis.set_title(self.Title)
        self.PLT.xlabel(self.xTitle, size = self.LabelSize)
        self.PLT.ylabel(self.yTitle, size = self.LabelSize)

        self.PLT.tight_layout()
        self.PLT.savefig(self.Filename, dpi = self.DPI)
        self.ChangeDirToRoot()
        self.PLT.close("all")

    def Get(self, var):
        return getattr(self, var)

    def Set(self, var, val):
        setattr(self, var, val)

    def GetBinWidth(self, Dims):
        if self.Get(Dims + "Min") == None or self.Get(Dims + "Max") == None:
            return False
        d_max, d_min, d_bin = self.Get(Dims + "Max"), self.Get(Dims + "Min"), self.Get(Dims + "Bins")
        return float((d_max - d_min) / (d_bin-1))

    def DefineRange(self, Dims):

        x = self.Get(Dims + "Data")
        x = [i for i in x if math.isnan(i) == False and math.isinf(i) == False]
        self.Set(Dims + "Data", x)

        if self.Get(Dims + "Min") == None:
            self.Set(Dims + "Min", min(self.Get(Dims + "Data")))

        if self.Get(Dims + "Max") == None:
            self.Set(Dims + "Max", max(self.Get(Dims + "Data")))
        
        if self.Get(Dims + "Bins") == None:
            p = set(self.Get(Dims + "Data"))
            self.Set(Dims + "Bins", max(p) - min(p)+1)
        
        if self.Get(Dims + "Range") == None:
            self.Set(Dims + "Range", (self.Get(Dims + "Min"), self.Get(Dims + "Max")))

    def CenteringBins(self, Dims):
        wb = self.GetBinWidth(Dims)
        self.Set(Dims + "Range", (self.Get(Dims + "Min")- wb*0.5, self.Get(Dims + "Max") + wb*0.5))

    def ApplyRandomTexture(self):
        if self.RandomTexture:
            ptr = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
            random.shuffle(ptr)
            return ptr[0]
        return 
    
    def ApplyRandomColor(self, obj):
        color = next(self.Axis._get_lines.prop_cycler)["color"]
        if obj.Color == None:
            obj.Color = color
        if obj.Color in self.Colors:
            obj.Color = None
            self.ApplyRandomColor(obj)
        else:
            self.Colors.append(obj.Color)
            return 


