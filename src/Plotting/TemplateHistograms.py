from AnalysisTopGNN.IO import WriteDirectory
from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.Plotting import CommonFunctions, Settings
import numpy as np
import mplhep as hep

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

    def ApplyFormat(self):
        obj, err, legen = hep.histplot(self.NPHisto, 
                density = self.Normalize, 
                label = self.Title,
                binticks = True, 
                linewidth = 3,
                alpha = self.Alpha,
                edgecolor = self.Color, 
                color = self.Color, 
                histtype = self.FillHist, 
                hatch = self.ApplyRandomTexture())[0]

    def Compile(self, Reset = True):
        if Reset:
            self.ResetPLT()
            self.DefineStyle()
            self.ApplyToPLT()

        if len(self.xData) == 0:
            self.Warning("EMPTY DATA. SKIPPING!")
            return

        self.DefineRange("x")
        if self.xBinCentering:
            self.CenteringBins("x")

        self.NPHisto = np.histogram(self.xData, bins = self.xBins, range = self.xRange, weights = self.xWeights)
        self.ApplyFormat()
       
        if self.xBinCentering:
            self.Axis.set_xticks(self.xData)
       
class CombineTH1F(CommonFunctions, WriteDirectory, Settings):
    def __init__(self, **kargs):
        Settings.__init__(self)
        Notification.__init__(self)
        WriteDirectory.__init__(self)
        self.DefineAxis("x")
        self.DefineAxis("y")
        self.DefineAxisData("x")
        self.Histograms = []
        self.Colors = []
        self.Histogram = None
        self.Stack = False
        self.ApplyInput(kargs)
        self.Caller = "Combine-TH1F"
    
    def ConsistencyCheck(self):
        b, H = [], []
        if self.Histogram != None:
            H.append(self.Histogram)
        H += self.Histograms
        
        for i in H:
            b.append(i.xBins)
            self.xData += i.xData

        self.DefineRange("x")
        self.CenteringBins("x")
        
        for i in H:
            i.xBins = self.xBins
            i.xMin = self.xMin
            i.xMax = self.xMax
            self.ApplyRandomColor(i)
            i.Compile()

    def Compile(self):
        self.ConsistencyCheck()

        self.ResetPLT()
        self.DefineStyle()
        self.ApplyToPLT()
        
        Sum = 1
        if self.Normalize == "%" and self.Histogram != None:
            Sum = self.Histogram.NPHisto[0].sum()*0.01
            self.Normalize = False
        elif self.Normalize and self.Histogram != None: 
            Sum = self.Histogram.NPHisto[0].sum()

        if self.Histogram != None:
            self.Histogram.NPHisto = (self.Histogram.NPHisto[0] / Sum, self.Histogram.NPHisto[1])
            self.Histogram.ApplyFormat()

        Labels, Numpies = [], []
        for i in self.Histograms:
            Labels.append(i.Title)
            i.NPHisto = (i.NPHisto[0]/Sum, i.NPHisto[1])
            Numpies.append(i.NPHisto)
            if self.Stack:
                continue
            i.ApplyFormat()
        
        if self.Stack:
            hep.histplot(Numpies, 
                    density = self.Normalize, 
                    histtype = self.FillHist, 
                    alpha = self.Alpha, 
                    stack = self.Stack, 
                    label = Labels, 
                    binticks = True)

        self.PLT.legend()
        if self.Logarithmic:
            self.PLT.yscale("log")

        if self.xBinCentering:
            self.Axis.set_xticks(self.xData)
    
        
    


