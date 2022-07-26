import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import matplotlib
matplotlib.use("agg")
from Functions.IO.Files import WriteDirectory
from Functions.Tools.Alerting import Notification

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
        self.Alpha = 0.5
        self.Color = "Orange"
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
        setattr(self, Dim + "Data", None)
        setattr(self, Dim + "Bins", None)
        setattr(self, Dim + "Weights", None)

    def DefineAxis(self, Dim):
        setattr(self, Dim + "Min", None)
        setattr(self, Dim + "Max", None)
        setattr(self, Dim + "Title", None)
    
    def ApplyInput(self, args):
        for key, val in args.items():
            if key not in self.__dict__:
                continue
            self.__dict__[key] = val

    def SaveFigure(self, Dir = None):
        if Dir == None:
            Dir = self.OutputDirectory
        if Dir.endswith("/") == False:
            Dir += "/"
        if ".png" not in self.Filename:
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

    def DefineRange(self, Dims):
        if getattr(self, Dims + "Min") == None:
            setattr(self, Dims + "Min", min(getattr(self, Dims + "Data")))

        if getattr(self, Dims + "Max") == None:
            setattr(self, Dims + "Max", max(getattr(self, Dims + "Data")))
        self.Range = (getattr(self, Dims + "Min"), getattr(self, Dims + "Max"))


class TH1F(CommonFunctions, WriteDirectory, Settings):
    def __init__(self, **kargs):
        Settings.__init__(self)
        Notification.__init__(self)
        WriteDirectory.__init__(self)
        self.DefineAxis("x")
        self.DefineAxisData("x")
        self.DefineAxis("y")
        self.ApplyInput(kargs)
        self.Caller = "TH1F"
        self.Alpha = 1

    def Compile(self):
        self.ResetPLT()
        self.DefineStyle()
        self.ApplyToPLT()

        self.DefineRange("x")
        H = np.histogram(self.xData, bins = self.xBins, range = self.Range, weights = self.xWeights)
        obj, err, legen = hep.histplot(H, density = self.Normalize, color = self.Color, histtype = self.FillHist)[0]
        self.NPHisto = H
        
       
class CombineTH1F(CommonFunctions, WriteDirectory, Settings):
    def __init__(self, **kargs):
        Settings.__init__(self)
        Notification.__init__(self)
        WriteDirectory.__init__(self)
        self.DefineAxis("x")
        self.DefineAxis("y")
        self.DefineAxisData("x")
        self.ApplyInput(kargs)
        self.Caller = "Combine-TH1F"
        self.Histograms = []
        self.Stack = False

    def Compile(self):
        Labels = []
        Numpies = []
        for i in self.Histograms:
            i.Compile()
            Labels.append(i.Title)
            Numpies.append(i.NPHisto)
            self.xData.append(i.xMin)
            self.xData.append(i.xMax)

        self.ResetPLT()
        self.DefineStyle()
        self.ApplyToPLT()

        hep.histplot(Numpies, density = self.Normalize, histtype = self.FillHist, 
                alpha = self.Alpha, stack = self.Stack, label = Labels)

        self.PLT.legend()
        if self.Logarithmic:
            self.PLT.yscale("log")
        self.PLT.xlim(self.xMin, self.xMax)

    
        
    


