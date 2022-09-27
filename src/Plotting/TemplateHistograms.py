from AnalysisTopGNN.Plotting import CommonFunctions
import numpy as np
import mplhep as hep

class Functions(CommonFunctions):

    def __init__(self):
        CommonFunctions.__init__(self)

        # --- Histogram Cosmetic Styles --- #
        self.Texture = False
        self.Alpha = 0.5
        self.FillHist = "fill"
        
        # --- Data Display --- #
        self.Normalize = None    
    
    def DefineAxisBins(self, Dim):
        self.Set(Dim + "Bins", None)
        self.Set(Dim + "BinCentering", False)
        self.Set(Dim + "Range", None)

    def ApplyRandomTexture(self):
        if self.Texture:
            ptr = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
            random.shuffle(ptr)
            return ptr[0]
        return 

    def GetBinWidth(self, Dims):
        if self.Get(Dims + "Min") == None or self.Get(Dims + "Max") == None:
            return False
        d_max, d_min, d_bin = self.Get(Dims + "Max"), self.Get(Dims + "Min"), self.Get(Dims + "Bins")
        return float((d_max - d_min) / (d_bin-1))

    def CenteringBins(self, Dims):
        wb = self.GetBinWidth(Dims)
        self.Set(Dims + "Range", (self.Get(Dims + "Min")- wb*0.5, self.Get(Dims + "Max") + wb*0.5))
 
    def DefineRange(self, Dims):
        self.DefineCommonRange(Dims)

        if self.Get(Dims + "Bins") == None:
            p = set(self.Get(Dims + "Data"))
            self.Set(Dims + "Bins", max(p) - min(p)+1)
        
        if self.Get(Dims + "Range") == None:
            self.Set(Dims + "Range", (self.Get(Dims + "Min"), self.Get(Dims + "Max")))


class TH1F(Functions):
    def __init__(self, **kargs):
        Functions.__init__(self)

        self.DefineAxisData("x")
        self.DefineAxisBins("x")

        self.DefineAxisData("y")
        self.DefineAxisBins("y")

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
       
class CombineTH1F(Functions):
    def __init__(self, **kargs):
        Functions.__init__(self)

        self.DefineAxisData("x")
        self.DefineAxisBins("x")

        self.DefineAxisData("y")
        self.DefineAxisBins("y")

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
            i.xBinCentering = self.xBinCentering
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
    
            
class TH1FStack(CombineTH1F):

    def __init__(self, **kargs):
        self.Histogram = None
        self.Data = {}
        CombineTH1F.__init__(self, **kargs)
   
    def __Recursive(self, inpt, search):
        if isinstance(inpt, dict) == False:
            return inpt
        if search in inpt:
            if isinstance(inpt[search], list):
                return inpt[search]

            out = []
            for k in inpt[search]:
                out += [k]*inpt[search][k]
            return out
        return [l for i in inpt for l in self.__Recursive(inpt[i], search)]

    def __Organize(self):
        Hists = {}
        if self.Histogram != None:
            Hists |= { self.Histogram : None }
        try:
            Hists |= { key : None for key in self.Histograms }
        except TypeError:
            Hists = self.Histograms
        self.Histograms = {}
        for i in Hists:
            if isinstance(i, str):
                params = {"xData" : self.__Recursive(self.Data, i), "Title" : i}
            else:
                params = i
                i = params["Title"]
            self.Histograms[i] = params
    
    def Precompiler(self):
        self.__Organize()
        hists = []
        for i in self.Histograms:
            if self.Histogram == i:
                self.Histogram = TH1F(**self.Histograms[i])
            else:
                hists.append(TH1F(**self.Histograms[i]))
        self.Histograms = hists



