from AnalysisTopGNN.Plotting import Settings, CommonFunctions
from AnalysisTopGNN.IO import WriteDirectory
from AnalysisTopGNN.Tools import Notification
import math
import statistics

class CommonFunctions(CommonFunctions, Settings, WriteDirectory):
    def __init__(self):
        Notification.__init__(self)
        WriteDirectory.__init__(self)

        # --- Element Sizes ---
        self.FontSize = 10
        self.LabelSize = 12.5
        self.TitleSize = 10
        self.LegendSize = 10

        # --- Figure Settings --- #
        self.Scaling = 1.25
        self.DPI = 250
        
        # --- Cosmetic --- #
        self.Style = None
        self.ATLASData = False
        self.ATLASYear = None
        self.ATLASLumi = None
        self.ATLASCom = None
        self.LineStyle = None
        self.Color = None
        self.Marker = None
        self.Label = None
 
        # --- Line Properties --- #
        self.Title = None
        self.DefineAxis("x")
        self.DefineAxis("y")
        self.DefineAxis("up_y")
        self.DefineAxis("down_y")
        self.ErrorBars = False
        self.Logarithmic = None
        self.DoStatistics = False
        
        self.OutputDirectory = "Plots"
        self.Filename = None
   
    def DefineAxis(self, Dim):
        self.Set(Dim + "Min", None)
        self.Set(Dim + "Max", None)
        self.Set(Dim + "Title", None)
        self.Set(Dim + "Data", [])   

    def DefineRange(self, Dims):
        
        if len(self.Get(Dims + "Data")) == 0:
            return 

        if self.Get(Dims + "Min") == None:
            self.Set(Dims + "Min", min(self.Get(Dims + "Data"))*0.9)

        if self.Get(Dims + "Max") == None:
            self.Set(Dims + "Max", max(self.Get(Dims + "Data"))*1.1)

    def ApplyRandomTexture(self, obj):
        ptr = ["-" , "--", "-.", ":" ]
        if obj.LineStyle == True:
            random.shuffle(ptr)
        elif obj.LineStyle != None:
            return 
        obj.LineStyle = ptr[0] 

    def ApplyRandomMarker(self, obj):
        ptr = ["x", "o", "O"]
        if obj.Marker == True:
            random.shuffle(ptr)
        elif obj.Marker != None:
            return 
        obj.Marker = ptr[0] 

    def GetBinWidth(self):
        pass
    
    def CenteringBins(self):
        pass

    def SanitizeData(self, var):
        out = []
        for i in range(len(var)):
            if var[i] == None:
                out.append(i) 
                continue
            if math.isinf(var[i]): 
                out.append(i)
                continue
            if math.isnan(var[i]):
                out.append(i)
                continue
        return out 
    
    def MakeStatistics(self, inpt):
        if isinstance(inpt, dict):
            xinpt = list(inpt.keys())
            inpt = list(inpt.values())
        else:
            xinpt = [i+1 for i in range(len(inpt))]            

        output = {}
        for i in range(len(inpt)):
            mean = statistics.mean(inpt[i])
            stdev = statistics.pstdev(inpt[i], mean)/math.sqrt(len(inpt[i]))
            output[xinpt[i]] = [mean, stdev]
        return output

class TLine(CommonFunctions):
    
    def __init__(self, **kargs):
        CommonFunctions.__init__(self)

        self.ApplyInput(kargs)
        self.Caller = "TLINE"
 
    def Compile(self, Reset = True):
        if Reset:
            self.ResetPLT()
            self.DefineStyle()
            self.ApplyToPLT()

        self.ApplyRandomMarker(self)

        if self.DoStatistics:
            self._temp = self.xData
            out = self.MakeStatistics(self.xData)
            self.xData = [i for i in out]
            self.yData = [out[i][0] for i in out]
            self.up_yData = [out[i][1] for i in out]
            self.down_yData = [out[i][1] for i in out]

        out = self.SanitizeData(self.xData)
        out += self.SanitizeData(self.yData)
        out = list(set(out))

        er_out = self.SanitizeData(self.down_yData)
        er_out += self.SanitizeData(self.up_yData)
        er_out = list(set(er_out)) 

        self.xData = [self.xData[i] for i in range(len(self.xData)) if i not in out]
        self.yData = [self.yData[i] for i in range(len(self.yData)) if i not in out]      
        self.up_yData = [self.up_yData[i] for i in range(len(self.up_yData)) if i not in out]
        self.down_yData = [self.down_yData[i] for i in range(len(self.down_yData)) if i not in out]

        self.DefineRange("y")
        self.DefineRange("x")
        
        if len(self.xData) != len(self.yData):
            return 

        if len(self.up_yData) != 0 and len(self.down_yData) != 0:
            self.PLT.errorbar(x = self.xData, y = self.yData, yerr = [self.up_yData, self.down_yData], 
                        linestyle = self.LineStyle, color = self.Color, marker = self.Marker, 
                        linewidth = 1, capsize = 3, label = self.Label) 
        else:
            self.PLT.plot(self.xData, self.yData, marker = self.Marker, color = self.Color, linewidth = 1, label = self.Label)
        
        if Reset:
            self.PLT.title(self.Title)
            self.PLT.xlabel(self.xTitle)
            self.PLT.ylabel(self.yTitle)
            self.PLT.xlim(self.xMin, self.xMax)
            self.PLT.ylim(self.yMin, self.yMax)
        if self.Logarithmic:
            self.PLT.yscale("log")
        if self.DoStatistics:
            self.xData = self._temp

class CombineTLine(CommonFunctions):

    def __init__(self, **kwargs):
        CommonFunctions.__init__(self)

        self.Lines = []
        self.Colors = []
        
        self.ApplyInput(kwargs)
        self.Caller = "COMBINED-TLINE"
        self.ResetPLT()
        

    def ConsistencyCheck(self):

        for i in self.Lines:
            self.yData += i.yData
            self.xData += i.xData
 
        out = self.SanitizeData(self.xData)
        out += self.SanitizeData(self.yData)
        out = list(set(out))
        
        self.xData = [self.xData[i] for i in range(len(self.xData)) if i not in out]
        self.yData = [self.yData[i] for i in range(len(self.yData)) if i not in out]   

        self.DefineRange("x")
        self.DefineRange("y")
        
        for i in self.Lines:
                
            i.xMin = self.xMin
            i.xMax = self.xMax
            i.yMin = self.yMin
            i.yMax = self.yMax

            if self.xTitle == None:
                self.xTitle = i.xTitle
            if self.yTitle == None:
                self.yTitle = i.yTitle
            if i.Label == None:
                i.Label = i.Title

            self.ApplyRandomColor(i)
            self.ApplyRandomMarker(i)
            i.Compile()

    def Compile(self):

        self.ConsistencyCheck()

        self.ResetPLT()
        self.DefineStyle()
        self.ApplyToPLT()
       
        for i in self.Lines:
            i.Compile(False)

        self.PLT.legend()
        if self.Logarithmic:
            self.PLT.yscale("log")

        self.PLT.title(self.Title)
        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)
        self.PLT.xlim(self.xMin, self.xMax)
        self.PLT.ylim(self.yMin, self.yMax)
